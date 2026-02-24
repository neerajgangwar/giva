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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.other import transpose

from .._buffer_dict import BufferDict


class OsoraLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ("osora_lambda_b", "osora_lambda_d")
    other_param_names = ("osora_A", "osora_B")

    def __init__(self, base_layer: nn.Module, **kwargs):
        self.base_layer = base_layer
        self.r = {}
        self.osora_dropout = nn.ModuleDict({})

        # For storing vector scale
        self.osora_lambda_b = nn.ParameterDict({})
        self.osora_lambda_d = nn.ParameterDict({})

        # Stores a reference to the osora_A/B BufferDict.
        # Set to `None` otherwise to avoid computation with random weights
        self.osora_A: Optional[BufferDict] = BufferDict({})
        self.osora_B: Optional[BufferDict] = BufferDict({})

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
        adapter_name,
        r,
        osora_dropout,
        init_weights,
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        r = min(r, min(self.in_features, self.out_features))
        self.r[adapter_name] = r
        if osora_dropout > 0.0:
            osora_dropout_layer = nn.Dropout(p=osora_dropout)
        else:
            osora_dropout_layer = nn.Identity()

        self.osora_dropout.update(nn.ModuleDict({adapter_name: osora_dropout_layer}))
        # Actual trainable parameters
        self.osora_lambda_b[adapter_name] = nn.Parameter(torch.ones(self.out_features), requires_grad=True)
        self.osora_lambda_d[adapter_name] = nn.Parameter(torch.randn(r), requires_grad=True)

        # non trainable references to osora_A/B buffers
        self.osora_A[adapter_name] = torch.randn(r, self.in_features, requires_grad=False)
        self.osora_B[adapter_name] = torch.randn(self.out_features, r, requires_grad=False)

        if init_weights:
            self.reset_osora_parameters(adapter_name)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    @torch.no_grad()
    def reset_osora_parameters(self, adapter_name):
        if adapter_name in self.osora_lambda_b.keys():
            nn.init.ones_(self.osora_lambda_b[adapter_name])

        weight = self.get_base_layer().weight
        device = weight.device
        dtype = weight.dtype
        r = self.r[adapter_name]
        # TODO: the device is hardcoded to speed up the computations
        # U, S, V = torch.svd_lowrank(weight.to(torch.float32).to('cuda'), q=min(4 * r, min(weight.size())), niter=4)
        U, S, Vh = torch.linalg.svd(weight.to(torch.float32).to('cuda'), full_matrices=False)
        V = Vh.T

        B = U[:, :r]
        A = V[:, :r].T
        S = S[:r]

        assert self.osora_A[adapter_name].data.size() == A.size(), (self.osora_A[adapter_name].data.size(), A.size())
        assert self.osora_B[adapter_name].data.size() == B.size(), (self.osora_B[adapter_name].data.size(), B.size())
        self.osora_A[adapter_name].data = A.contiguous()
        self.osora_B[adapter_name].data = B.contiguous()
        self.osora_lambda_d[adapter_name].data = S

        offset = B @ (S.unsqueeze(-1) * A)
        self.get_base_layer().weight.data -= offset.to(dtype).to(device)


class Linear(nn.Linear, OsoraLayer):
    # Osora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        osora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_weights: bool = True,
        **kwargs,
    ) -> None:
        # this gets the init from nn.Linear's super perspective, i.e. nn.Module.__init__, which should always be called
        super(nn.Linear, self).__init__()
        OsoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, osora_dropout, init_weights)
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
            if active_adapter in self.osora_lambda_d.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()

                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.osora_lambda_d.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        osora_A = self.osora_A[adapter]
        osora_B = self.osora_B[adapter]

        device = osora_B.device
        dtype = osora_B.dtype

        # In case users wants to merge the adapter weights that are in
        # (b)float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # (b)float16 because some CPUs have slow bf16/fp16 matmuls.
        cast_to_fp32 = device.type == "cpu" and (dtype == torch.float16 or dtype == torch.bfloat16)

        lambda_d = self.osora_lambda_d[adapter]
        lambda_b = self.osora_lambda_b[adapter]

        if cast_to_fp32:
            osora_A = osora_A.float()
            osora_B = osora_B.float()
            lambda_d = lambda_d.float()
            lambda_b = lambda_b.float()

        sliced_A = osora_A.to(lambda_d.device)
        sliced_B = osora_B.to(lambda_d.device)
        lambda_b = lambda_b.unsqueeze(-1)
        lambda_d = lambda_d.unsqueeze(-1)
        output_tensor = transpose((lambda_b * sliced_B) @ (lambda_d * sliced_A), self.fan_in_fan_out)

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

        return output_tensor

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.osora_lambda_d.keys():
                    continue

                lambda_d = self.osora_lambda_d[active_adapter]
                lambda_b = self.osora_lambda_b[active_adapter]

                osora_A = self.osora_A[active_adapter]
                osora_B = self.osora_B[active_adapter]

                dropout = self.osora_dropout[active_adapter]
                x = x.to(lambda_d.dtype)
                result = result + lambda_b * F.linear(lambda_d * F.linear(dropout(x), osora_A), osora_B)

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "osora." + rep
