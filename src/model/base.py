import logging
import torch
import lightning as L
from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from omegaconf import DictConfig, OmegaConf
from peft import (
    get_peft_model,
    LoraConfig,
    VeraConfig,
    RandLoraConfig,
    GivaConfig,
    OsoraConfig,
    GradientContext,
)


logger = logging.getLogger(__name__)


class BaseModel(L.LightningModule, ABC):
    def __init__(
        self,
        model_name_or_path: str,
        optim: str,
        lr: float,
        weight_decay: float,
        training_mode: DictConfig,
        scheduler_type: str,
        warmup_steps: float,
        task_type: str,
        max_input_tokens: int=1024,
        peft_kwargs: Dict[str, Any]={},
    ):
        super(BaseModel, self).__init__()

        self.model_name_or_path = model_name_or_path
        self.optim = optim
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_mode = training_mode
        self.scheduler_type = scheduler_type
        self.warmup_steps = warmup_steps
        self.max_input_tokens = max_input_tokens
        self.task_type = task_type
        self.peft_kwargs = peft_kwargs

        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        self.model.train()
        self.named_grads = None
        self.adapters_added = False

        self.save_hyperparameters()


    @abstractmethod
    def get_tokenizer(self):
        pass


    @abstractmethod
    def get_model(self):
        pass


    @abstractmethod
    def collate_fn(self, batch: Any, is_test: bool=True) -> Dict[str, Any]:
        pass


    def set_named_gradients(self, named_grads: Dict[str, torch.Tensor]) -> None:
        self.named_grads = named_grads


    def add_adapters(self):
        if self.adapters_added:
            return

        if self.training_mode.type == 'lora':
            assert self.training_mode.config is not None
            with GradientContext(self.model, self.named_grads if self.training_mode.init_type == 'gradient' else None):
                self.model = get_peft_model(
                    self.model,
                    LoraConfig(**OmegaConf.to_container(self.training_mode.config), task_type=self.task_type, **self.peft_kwargs),
                )
        elif self.training_mode.type == 'vera':
            assert self.training_mode.config is not None
            self.model = get_peft_model(
                self.model,
                VeraConfig(**OmegaConf.to_container(self.training_mode.config), task_type=self.task_type, **self.peft_kwargs),
            )
        elif self.training_mode.type == 'osora':
            assert self.training_mode.config is not None
            self.model = get_peft_model(
                self.model,
                OsoraConfig(**OmegaConf.to_container(self.training_mode.config), task_type=self.task_type, **self.peft_kwargs),
            )
        elif self.training_mode.type == 'randlora':
            assert self.training_mode.config is not None
            self.model = get_peft_model(
                self.model,
                RandLoraConfig(**OmegaConf.to_container(self.training_mode.config), task_type=self.task_type, **self.peft_kwargs),
            )
        elif self.training_mode.type == 'giva':
            assert self.training_mode.config is not None
            with GradientContext(self.model, self.named_grads):
                self.model = get_peft_model(
                    self.model,
                    GivaConfig(**OmegaConf.to_container(self.training_mode.config), task_type=self.task_type, **self.peft_kwargs),
                )

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Total params: {total_params}, Trainable params: {trainable_params}')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f'Trainable: {name}')

        self.adapters_added = True


    def configure_model(self):
        self.add_adapters()


    def configure_optimizers(self) -> Any:
        optimizer = self.create_optimizer()

        if self.scheduler_type is None or self.scheduler_type.lower() == 'none':
            return optimizer

        if self.warmup_steps <= 1:
            num_warmup_steps = int(self.trainer.estimated_stepping_batches * self.warmup_steps)
        else:
            num_warmup_steps = int(self.warmup_steps)

        logger.info(f'Using {num_warmup_steps} warmup steps')

        if self.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=num_warmup_steps,
            )
        elif self.scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            raise NotImplementedError(f'scheduler_type: {self.scheduler_type}')

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]


    def create_optimizer(self):
        if self.optim.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError(f'Optim: {self.optim}')
        return optimizer


    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)

        if self.training_mode.type == 'fft' or (self.training_mode.type == 'lora' and self.training_mode.init_type == 'gradient'):
            return

        trainable_layers = [name for name, param in self.named_parameters() if param.requires_grad]
        state = checkpoint['state_dict']
        for name in list(state.keys()):
            if name not in trainable_layers and self.training_mode.type not in name:
                state.pop(name)
