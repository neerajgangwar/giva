import torch
from typing import Callable, Optional
from typing import Dict, List
from tqdm import tqdm
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import PreTrainedModel


def get_device_with_meta_params(model: torch.nn.Module) -> torch.device:
    """
    Get the device of the model's parameters. Useful if some parameters are on meta device.
    """
    devices = list({p.device for p in model.parameters() if p.device.type != "meta"})
    if len(devices) > 1:
        raise Exception(f"Could not determine device, model has multiple devices: {devices}")
    return devices[0]


def estimate_gradient(
    model: PreTrainedModel,
    dataloader: DataLoader,
    compute_loss: Optional[Callable]=None,
    num_batches: int=-1,
    mixed_precision: str='bf16',
    use_gradient_checkpointing: bool=True,
) -> Dict[str, List[torch.Tensor]]:
    """
    Estimates the gradients of a model's parameters over a dataset.

    Args:
        model (PreTrainedModel): The model whose gradients will be estimated.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
        accelerator (Accelerator): The accelerator used for training.
        quant_type (str, optional): The data type for quantizing the model parameters. Defaults to "nf4".
        no_split_module_classes (list of type, optional): List of module classes that should not be split during offloading. Defaults to None.

    Returns:
        Dict[str, List[torch.Tensor]]: A dictionary mapping parameter names to their estimated gradients.
    """
    print(f'Using mixed_precision: {mixed_precision}')
    device = get_device_with_meta_params(model)
    training = model.training
    is_gradient_checkpointing = model.is_gradient_checkpointing

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    accelerator = Accelerator(mixed_precision=mixed_precision)
    model, dataloader = accelerator.prepare(model, dataloader)
    print(f'Running gradient estimation on {accelerator.device}')

    named_grads = {}
    model.train()
    model.zero_grad()
    for batch_idx, batch in tqdm(enumerate(dataloader), desc="Estimating gradient"):
        with accelerator.autocast():
            loss = compute_loss(model, batch)
            assert not torch.isnan(loss)
            accelerator.backward(loss)

        if num_batches != -1 and batch_idx >= num_batches:
            break

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None and name.endswith('.weight'):
            named_grads[name] = param.grad.cpu() / (len(dataloader) if num_batches == -1 else num_batches)

    named_grads = {".".join(k.split(".")[:-1]): v for k, v in named_grads.items()}

    # Reset model gradient and training model
    model.zero_grad()
    model.train(training)
    model.to(device)

    # Reset gradient checkpointing
    if use_gradient_checkpointing:
        if is_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()

    return named_grads


class GradientContext:
    """
    Context manager for attaching and detaching a named gradient dictionary to a model.

    This context manager allows you to temporarily attach a dictionary of named gradients
    to the model as an attribute. Upon entering the context, the `named_grad` dictionary
    is set as an attribute of the model. Upon exiting the context, the attribute is removed.

    Attributes:
        model (torch.nn.Module): The model to which the gradient dictionary will be attached.
        named_grad (dict, optional): A dictionary where keys are parameter names and values are gradients. Defaults to None.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        named_grad: dict = None,
    ) -> None:
        self.model = model
        self.named_grad = named_grad

    def __enter__(self):
        setattr(self.model, "named_grad", self.named_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.model, "named_grad"):
            delattr(self.model, "named_grad")
