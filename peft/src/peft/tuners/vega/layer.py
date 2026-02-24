# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Optional, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose
from .._buffer_dict import BufferDict


class VegaLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("vega_lambda_o", "vega_lambda_s")
    other_param_names = ("vega_A", "vega_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.vega_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.vega_lambda_o = nn.ParameterDict({})
        self.vega_lambda_s = nn.ParameterDict({})

        # Stores a reference to the vera_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.vega_A = BufferDict({}, persistent=True)
        self.vega_B = BufferDict({}, persistent=True)

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )

        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        vega_dropout: float,
        init_weights: str,
        s_initial: float,
        o_initial: float,
        fan_in_fan_out: bool,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        if init_weights == "VrU2r":
            r = min(r, min(self.in_features, self.out_features) // 2)
        elif init_weights in ("VrUr", "Vr", "VrUrand"):
            r = min(r, min(self.in_features, self.out_features))

        self.r[adapter_name] = r

        if vega_dropout > 0.0:
            vega_dropout_layer = nn.Dropout(p=vega_dropout)
        else:
            vega_dropout_layer = nn.Identity()

        self.vega_dropout.update(nn.ModuleDict({adapter_name: vega_dropout_layer}))

        self.vega_lambda_s[adapter_name] = nn.Parameter(torch.ones(r), requires_grad=True)
        self.vega_lambda_o[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)

        # Non-trainable
        self.vega_A[adapter_name] = torch.randn(r, self.in_features)
        self.vega_B[adapter_name] = torch.randn(self.out_features, r)

        if init_weights:
            self.init_A_B(
                adapter_name=adapter_name,
                init_weights=init_weights,
                s_initial=s_initial,
                o_initial=o_initial,
                fan_in_fan_out=fan_in_fan_out,
            )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def init_A_B(self, adapter_name: str, init_weights: str, s_initial: float, o_initial: float, fan_in_fan_out: bool) -> None:
        if "grad" not in self.kwargs.keys():
            return

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        device = weight.device
        dtype = weight.dtype

        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise NotImplementedError(f"For dtype: {dtype}")

        grad = self.kwargs["grad"].to(device).to(torch.float32)
        r = self.r[adapter_name]
        grad = transpose(grad, fan_in_fan_out)

        U, S, V = torch.svd_lowrank(grad, q=min(4 * r, min(grad.shape)), niter=4)
        if init_weights == "VrU2r":
            A = V[:, :r].T
            B = U[:, r:2 * r]
        elif init_weights == "VrUr":
            A = V[:, :r].T
            B = U[:, :r]
        elif init_weights == "Vr":
            A = V[:, :r].T
            B = torch.randn(U.size(0), r, device=device)
            B, _ = torch.linalg.qr(B, mode="reduced")
        elif init_weights == "VrUrand":
            A = V[:, :r].T
            B = torch.randn(U.size(0), r, device=device)
        else:
            raise NotImplementedError(f"For '{init_weights}'")

        self.vega_A[adapter_name].data = A.contiguous()
        self.vega_B[adapter_name].data = B.contiguous()

        if adapter_name in self.vega_lambda_s.keys():
            assert o_initial == 0, f'o_initial must be 0 but got "{o_initial}"'
            assert s_initial == 1, f's_initial must be 1 but got "{s_initial}'
            nn.init.zeros_(self.vega_lambda_s[adapter_name]).fill_(s_initial)
            nn.init.zeros_(self.vega_lambda_o[adapter_name]).fill_(o_initial)


class Linear(nn.Linear, VegaLayer):
    # Vega implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        vega_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: str = 'VrU2r',
        s_initial: float = 1.,
        o_initial: float = 0.,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        VegaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            vega_dropout=vega_dropout,
            init_weights=init_weights,
            s_initial=s_initial,
            o_initial=o_initial,
            fan_in_fan_out=fan_in_fan_out,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.vega_lambda_o.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weight = base_layer.weight.data.clone()
                    orig_dtype = orig_weight.dtype
                    delta_weight = self.get_delta_weight(active_adapter)
                    orig_weight += delta_weight.to(orig_dtype)

                    if not torch.isfinite(orig_weight).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weight
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)

                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.vega_lambda_o.keys():
                weight = self.get_base_layer().weight
                orig_dtype = weight.dtype
                delta_weight = self.get_delta_weight(active_adapter)
                weight.data -= delta_weight.to(orig_dtype)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        vega_A = self.vega_A[adapter]
        vega_B = self.vega_B[adapter]

        device = vega_B.device
        dtype = vega_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        lambda_o = self.vega_lambda_o[adapter]
        lambda_s = self.vega_lambda_s[adapter]

        if cast_to_fp32:
            vega_A = vega_A.float()
            vega_B = vega_B.float()
            lambda_s = lambda_s.float()
            lambda_o = lambda_o.float()

        lambda_s = lambda_s.unsqueeze(-1)
        lambda_o = lambda_o.unsqueeze(-1)
        output_tensor = transpose((lambda_o * vega_B) @ (lambda_s * vega_A), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            self.vega_A[adapter].weight.data = vega_A.to(dtype)
            self.vega_B[adapter].weight.data = vega_B.to(dtype)
            self.vega_lambda_s[adapter].weight.data = lambda_s.to(dtype)
            self.vega_lambda_o[adapter].weight.data = lambda_o.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        assert adapter_names is None

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            vega_A_keys = self.vega_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in vega_A_keys:
                    continue

                vega_A = self.vega_A[active_adapter]
                vega_B = self.vega_B[active_adapter]
                dropout = self.vega_dropout[active_adapter]
                lambda_s = self.vega_lambda_s[active_adapter]
                lambda_o = self.vega_lambda_o[active_adapter]
                x = self._cast_input_dtype(x, vega_A.dtype)
                result = result + lambda_o * F.linear(lambda_s * F.linear(dropout(x), vega_A), vega_B)

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vega." + rep


class ConcatLinear(Linear):
    # Vega implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        concat_enabled: List[bool],
        r: int = 0,
        vega_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: str = 'VrU2r',
        s_initial: float = 1.,
        o_initial: float = 0.,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        nn.Module.__init__(self).__init__()
        VegaLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out
        self.concat_enabled = {}
        self.enabled_indices = {}

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name=adapter_name,
            r=r,
            vega_dropout=vega_dropout,
            init_weights=init_weights,
            s_initial=s_initial,
            o_initial=o_initial,
            fan_in_fan_out=fan_in_fan_out,
            concat_enabled=concat_enabled,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def update_layer(
        self,
        adapter_name: str,
        r: int,
        vega_dropout: float,
        init_weights: str,
        s_initial: float,
        o_initial: float,
        fan_in_fan_out: bool,
        concat_enabled: List[bool],
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        if init_weights == "VrU2r":
            r = min(r, min(self.in_features, self.out_features) // 2)
        elif init_weights in ("VrUr", "Vr", "VrUrand"):
            r = min(r, min(self.in_features, self.out_features))

        self.r[adapter_name] = r

        if vega_dropout > 0.0:
            vega_dropout_layer = nn.Dropout(p=vega_dropout)
        else:
            vega_dropout_layer = nn.Identity()

        self.vega_dropout.update(nn.ModuleDict({adapter_name: vega_dropout_layer}))

        # For concat linear
        assert self.out_features % len(concat_enabled) == 0
        self.concat_enabled[adapter_name] = concat_enabled
        enabled_indices = torch.zeros((self.out_features, ), dtype=torch.bool).view(len(concat_enabled), -1)
        enabled_indices[concat_enabled, :] = True
        enabled_indices = enabled_indices.view(-1)
        self.enabled_indices[adapter_name] = enabled_indices

        self.vega_lambda_s[adapter_name] = nn.Parameter(torch.ones(r * sum(concat_enabled)), requires_grad=True)
        self.vega_lambda_o[adapter_name] = nn.Parameter(torch.ones(self.out_features * sum(concat_enabled) // len(concat_enabled)), requires_grad=True)

        # Non-trainable
        self.vega_A[adapter_name] = torch.randn(r * sum(concat_enabled), self.in_features)
        self.vega_B[adapter_name] = torch.randn(self.out_features * sum(concat_enabled) // len(concat_enabled), r)

        if init_weights:
            self.init_A_B(
                adapter_name=adapter_name,
                init_weights=init_weights,
                s_initial=s_initial,
                o_initial=o_initial,
                fan_in_fan_out=fan_in_fan_out,
                concat_enabled=concat_enabled,
            )

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def init_A_B(self, adapter_name: str, init_weights: str, s_initial: float, o_initial: float, fan_in_fan_out: bool, concat_enabled: List[bool]) -> None:
        if "grad" not in self.kwargs.keys():
            return

        assert o_initial == 0, f"o_initial must be 0 but {o_initial} provided"
        assert s_initial == 1, f"s_initial must be 1 but {s_initial} provided"

        base_layer = self.get_base_layer()
        weight = base_layer.weight
        device = weight.device
        dtype = weight.dtype

        if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
            raise NotImplementedError(f"For dtype: {dtype}")

        grad = self.kwargs["grad"].to(device).to(torch.float32)
        r = self.r[adapter_name]
        grad = transpose(grad, fan_in_fan_out)
        assert grad.size() == (self.out_features, self.in_features)
        grad = grad.reshape(len(concat_enabled), -1, grad.size(-1))
        grad = grad[concat_enabled]

        # U, S, V = torch.svd_lowrank(grad, q=min(4 * r, min(self.in_features, self.out_features)), niter=4)
        U, S, Vh = torch.linalg.svd(grad, full_matrices=False)
        V = Vh.transpose(-1, -2)
        if init_weights == "VrU2r":
            A = V[..., :r].transpose(1, 2)
            B = U[..., r:2 * r]
        elif init_weights == "VrUr":
            A = V[..., :r].transpose(1, 2)
            B = U[..., :r]
        elif init_weights == "Vr":
            A = V[..., :r].transpose(1, 2)
            B = torch.randn(*U.shape[:-1], r, device=device)
            B, _ = torch.linalg.qr(B, mode="reduced")
        elif init_weights == "VrUrand":
            A = V[..., :r].T
            B = torch.randn(*U.shape[:-1], r, device=device)
        else:
            raise NotImplementedError(f"For '{init_weights}'")

        assert A.size() == (sum(concat_enabled), r, self.in_features)
        assert B.size() == (sum(concat_enabled), self.out_features // len(concat_enabled), r)
        A = A.reshape(r * sum(concat_enabled), self.in_features)
        B = B.reshape(self.out_features * sum(concat_enabled) // len(concat_enabled), r)

        assert self.vega_A[adapter_name].data.size() == A.size()
        assert self.vega_B[adapter_name].data.size() == B.size()

        self.vega_A[adapter_name].data = A.contiguous()
        self.vega_B[adapter_name].data = B.contiguous()

        if adapter_name in self.vega_lambda_s.keys():
            nn.init.zeros_(self.vega_lambda_s[adapter_name]).fill_(s_initial)
            nn.init.zeros_(self.vega_lambda_o[adapter_name]).fill_(o_initial)

    def zero_pad(self, x: torch.Tensor, enabled_indices: torch.Tensor) -> torch.Tensor:
        result = x.new_zeros((len(enabled_indices), *x.shape[1:]))
        result[enabled_indices] = x
        return result

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        vega_A = self.vega_A[adapter]
        vega_B = self.vega_B[adapter]
        enabled_indices = self.enabled_indices[adapter]
        concat_enabled = self.concat_enabled[adapter]

        device = vega_B.device
        dtype = vega_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        lambda_o = self.vega_lambda_o[adapter]
        lambda_s = self.vega_lambda_s[adapter]


        if cast_to_fp32:
            vega_A = vega_A.float()
            vega_B = vega_B.float()
            lambda_s = lambda_s.float()
            lambda_o = lambda_o.float()

        lambda_s = lambda_s.unsqueeze(-1)
        lambda_o = lambda_o.unsqueeze(-1)
        weight_A = lambda_s * vega_A
        weight_B = lambda_o * vega_B
        output_tensor = F.conv1d(weight_A.unsqueeze(0), weight_B.unsqueeze(-1), groups=sum(concat_enabled)).squeeze(0)
        output_tensor = self.zero_pad(output_tensor, enabled_indices)
        output_tensor = transpose(output_tensor, self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            self.vega_A[adapter].weight.data = vega_A.to(dtype)
            self.vega_B[adapter].weight.data = vega_B.to(dtype)
            self.vega_lambda_s[adapter].weight.data = lambda_s.to(dtype)
            self.vega_lambda_o[adapter].weight.data = lambda_o.to(dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        assert adapter_names is None

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            vega_A_keys = self.vega_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in vega_A_keys:
                    continue

                vega_A = self.vega_A[active_adapter]
                dropout = self.vega_dropout[active_adapter]
                x = self._cast_input_dtype(x, vega_A.dtype)
                delta_weight = transpose(self.get_delta_weight(active_adapter), self.fan_in_fan_out)
                vega_output = F.linear(dropout(x), delta_weight)
                result = result + vega_output

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "vega." + rep

