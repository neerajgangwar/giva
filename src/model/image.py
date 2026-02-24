import logging
import torch
import torchvision
import numpy as np
from typing import Dict, Any
from transformers import (
    AutoConfig,
    CLIPProcessor,
    AutoImageProcessor,
    AutoModelForImageClassification,
    CLIPModel,
)
from omegaconf import DictConfig
from .base import BaseModel


logger = logging.getLogger(__name__)


class ImageClassificationModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str,
        optim: str,
        lr: float,
        weight_decay: float,
        training_mode: DictConfig,
        num_labels: int,
        scheduler_type: str,
        warmup_steps: float,
    ):
        self.num_labels = num_labels
        super(ImageClassificationModel, self).__init__(
            model_name_or_path=model_name_or_path,
            optim=optim,
            lr=lr,
            weight_decay=weight_decay,
            training_mode=training_mode,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            task_type=None,
            max_input_tokens=None,
            peft_kwargs={'modules_to_save': ['classifier']},
        )
        self.train_transformers = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ])


    def get_model(self):
        config = AutoConfig.from_pretrained(self.model_name_or_path, num_labels=self.num_labels)
        return AutoModelForImageClassification.from_pretrained(self.model_name_or_path, config=config)


    def get_tokenizer(self):
        return AutoImageProcessor.from_pretrained(self.model_name_or_path)


    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        batch_size = batch['pixel_values'].size(0)
        output = self.model(
            pixel_values=batch['pixel_values'],
            labels=batch['labels'],
            output_attentions=False,
            output_hidden_states=False,
        )
        loss = output['loss']
        self.log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def compute_accuracy(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(
            pixel_values=batch['pixel_values'],
            output_attentions=False,
            output_hidden_states=False,
        )
        logits = output['logits']
        preds = logits.argmax(dim=-1)
        correct = (preds == batch['labels']).sum().to(torch.float32)
        return correct / batch['labels'].size(0)


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        batch_size = batch['pixel_values'].size(0)
        accuracy = self.compute_accuracy(batch)
        self.log('val/dataset_size', batch_size, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='sum')
        self.log('val/accuracy', accuracy, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='mean')


    def on_test_epoch_start(self):
        if self.training_mode.type != 'fft':
            self.model.merge_and_unload(safe_merge=True)
        return super().on_test_epoch_start()


    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        batch_size = batch['pixel_values'].size(0)
        accuracy = self.compute_accuracy(batch)
        self.log('test/dataset_size', batch_size, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='sum')
        self.log('test/accuracy', accuracy, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='mean')


    def collate_fn(self, batch: Any, is_test: bool=True):
        imgs = [e['img'] for e in batch]
        labels = [e['label'] for e in batch]

        if is_test:
            processed_imgs = self.tokenizer.preprocess(imgs)
        else:
            processed_imgs = [self.train_transformers(img) for img in imgs]
            processed_imgs = self.tokenizer.preprocess(processed_imgs, do_resize=False, do_center_crop=False)

        labels = torch.tensor(labels, dtype=torch.long)
        processed_imgs = torch.tensor(np.array(processed_imgs['pixel_values']), dtype=torch.float32)

        return {
            'pixel_values': processed_imgs,
            'labels': labels,
        }
