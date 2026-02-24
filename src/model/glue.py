import logging
import torch
import evaluate
from typing import Dict, Any, Iterable, Optional
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
)
from omegaconf import DictConfig
from src.data.glue import EVAL_METRICS
from src.utils.glue import tokenize_multipart_input, LABEL_MAPPING, tokenize_label
from src.utils.common import create_padded_tensor
from .base import BaseModel


logger = logging.getLogger(__name__)


class GlueClassificationModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str,
        optim: str,
        lr: float,
        weight_decay: float,
        max_input_len: int,
        training_mode: DictConfig,
        num_labels: int,
        scheduler_type: str,
        warmup_steps: float,
    ):
        self.num_labels = num_labels
        super(GlueClassificationModel, self).__init__(
            model_name_or_path=model_name_or_path,
            optim=optim,
            lr=lr,
            weight_decay=weight_decay,
            training_mode=training_mode,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            task_type='SEQ_CLS',
            max_input_tokens=max_input_len,
        )


    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name_or_path)


    def get_model(self):
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=self.num_labels,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            config=config,
        )
        return model


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
        )
        loss = output['loss']
        self.log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def on_validation_epoch_start(self):
        self.preds = []
        self.labels = []
        self.task = None
        return super().on_validation_epoch_start()


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        task = batch['task']
        if self.task is None:
            self.task = task
        else:
            assert self.task == task

        preds = self.compute_predictions(batch)
        self.preds.extend(preds)
        self.labels.extend(batch['targets'])


    def on_validation_epoch_end(self):
        metric = evaluate.load(EVAL_METRICS[self.task])
        score = metric.compute(predictions=self.preds, references=self.labels)

        for key, val in score.items():
            self.log(f'val/{key}', val, on_epoch=True, sync_dist=True)
        self.log(f'val/dataset_size', len(self.preds), on_epoch=True, sync_dist=True)
        del self.preds
        del self.labels
        del self.task
        return super().on_validation_epoch_end()


    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        raise NotImplementedError


    def compute_predictions(self, batch: Dict[str, Any]) -> Iterable:
        task = batch['task']

        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
        )
        logits = output['logits']
        if task in ('stsb'):
            assert logits.size(1) == 1
            preds = logits.squeeze(1)
        else:
            preds = logits.argmax(dim=-1)

        return preds.detach().cpu().tolist()


    def collate_fn(self, batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        task = set(e['task'] for e in batch)
        assert len(task) == 1, f'Batch with multiple tasks {task}'
        task = list(task)[0]

        labels = [e['label'] for e in batch]

        if task == 'stsb':
            labels = torch.tensor(labels, dtype=torch.float)
        else:
            labels = torch.tensor(labels, dtype=torch.long)

        if task in ('sst2', 'cola'):
            sentences = [e['sentence'] for e in batch]
            tokenized = self.tokenizer(
                sentences,
                padding=True,
                max_length=self.max_input_tokens,
                truncation=True,
                return_tensors='pt',
            )
        elif task == 'qnli':
            questions = [e['question'] for e in batch]
            sentences = [e['sentence'] for e in batch]
            tokenized = self.tokenizer(
                questions,
                sentences,
                padding=True,
                max_length=self.max_input_tokens,
                truncation=True,
                return_tensors='pt',
            )
        elif task in ('mrpc', 'rte', 'stsb'):
            sentences1 = [e['sentence1'] for e in batch]
            sentences2 = [e['sentence2'] for e in batch]
            tokenized = self.tokenizer(
                sentences1,
                sentences2,
                padding=True,
                max_length=self.max_input_tokens,
                truncation=True,
                return_tensors='pt',
            )
        else:
            raise NotImplementedError(f'Task: {task}')

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels,
            'targets': labels.cpu().tolist(),
            'task': task,
        }
