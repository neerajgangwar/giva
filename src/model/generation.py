import logging
import torch
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from omegaconf import DictConfig
from src.utils.commonsense import extract_answer as extract_answer_commonsense
from src.utils.math import extract_answer_pred as extract_answer_gsm8k
from src.utils.code import post_process as post_process_code
from src.utils.common import create_padded_tensor
from .base import BaseModel


logger = logging.getLogger(__name__)


class GenerationModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str,
        optim: str,
        lr: float,
        weight_decay: float,
        training_mode: DictConfig,
        scheduler_type: str,
        warmup_steps: float,
        max_new_tokens: int,
        max_input_tokens: int=1024,
        train_on_inputs: bool=False,
        num_beams: int=1,
    ):
        super(GenerationModel, self).__init__(
            model_name_or_path=model_name_or_path,
            optim=optim,
            lr=lr,
            weight_decay=weight_decay,
            training_mode=training_mode,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            max_input_tokens=max_input_tokens,
            task_type='CAUSAL_LM',
        )
        self.max_new_tokens = max_new_tokens
        self.train_on_inputs = train_on_inputs
        self.num_beams = num_beams


    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        config = AutoConfig.from_pretrained(self.model_name_or_path)
        tokenizer.padding_side = 'left'

        assert tokenizer.eos_token and tokenizer.eos_token_id
        assert tokenizer.eos_token == tokenizer.convert_ids_to_tokens(config.eos_token_id)
        assert tokenizer.eos_token_id == config.eos_token_id

        if tokenizer.bos_token:
            assert tokenizer.bos_token_id
        else:
            assert not tokenizer.bos_token_id
            tokenizer.bos_token_id = config.bos_token_id if config.bos_token_id else tokenizer.eos_token_id
            bos_token = tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id)
            assert isinstance(bos_token, str)
            tokenizer.bos_token = bos_token

        if tokenizer.pad_token:
            assert tokenizer.pad_token_id
        else:
            assert not tokenizer.pad_token_id
            tokenizer.pad_token_id = config.pad_token_id if config.pad_token_id else tokenizer.eos_token_id
            pad_token = tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id)
            assert isinstance(pad_token, str)
            tokenizer.pad_token = pad_token

        return tokenizer


    def get_model(self):
        return AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.bfloat16)


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],
            labels=batch['labels'],
        )
        loss = output['loss']
        self.log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=True)
        return loss


    def compute_accuracy(self, batch: Dict[str, str]) -> Dict[str, float]:
        # Do not pass position_ids during generation as it needs to be updated after every token
        # generation
        output = self.model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
        )
        input_texts = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        input_len = batch['input_ids'].size(1)
        output = output[:, input_len:]
        pred_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        total, correct = 0, 0
        for batch_idx, (input_text, dataset, pred, target) in enumerate(zip(input_texts, batch['datasets'], pred_texts, batch['targets'])):
            if dataset == 'gsm8k':
                pred_answer = extract_answer_gsm8k(pred)
            elif dataset == 'humaneval':
                pred_answer = post_process_code(pred)
            else:
                pred_answer = extract_answer_commonsense(dataset, pred)

            total += 1
            if target is not None:
                correct += int(pred_answer == target)

            model_output = {
                'input_text': input_text,
                'pred': pred,
                'pred_answer': pred_answer,
                'target': target,
            }
            if 'task_ids' in batch:
                model_output['task_id'] = batch['task_ids'][batch_idx]

            self.model_outputs.append(model_output)

        accuracy = correct / total
        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
        }


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            position_ids=batch['position_ids'],
            labels=batch['labels'],
        )
        loss = output['loss']
        self.log('val/loss', loss, batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        return loss


    def on_test_epoch_start(self):
        self.model_outputs = []
        self.model.merge_and_unload(safe_merge=True)
        return super().on_test_epoch_start()


    def test_step(self, batch: Dict[str, str], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        accuracy = self.compute_accuracy(batch)
        self.log('test/accuracy', accuracy['accuracy'], batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/dataset_size', accuracy['total'], batch_size=batch_size, on_step=False, on_epoch=True, sync_dist=True, reduce_fx='sum')


    def collate_fn(self, batch: Any, is_test: bool=True) -> Dict[str, Any]:
        input_texts = [e['input_text'] for e in batch]
        output_texts = [e['output_text'] for e in batch]
        targets = [e['target'] for e in batch]
        datasets = [e['dataset'] for e in batch]
        task_ids = [e.get('task_id', None) for e in batch]

        input_ids, attn_mask, labels = [], [], []
        for input_text, output_text in zip(input_texts, output_texts):
            if not is_test:
                text = f'{input_text}{output_text}'
                curr_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(text, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
                input_len = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids']) + 1 # +1 for the bos token
                curr_labels = curr_input_ids.copy()
                if not self.train_on_inputs:
                    curr_labels[:input_len] = [-100] * input_len
            else:
                text = f'{input_text}'
                curr_input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(text, add_special_tokens=False)['input_ids']
                curr_labels = [-100] * len(curr_input_ids)

            input_ids.append(curr_input_ids)
            labels.append(curr_labels)
            attn_mask.append([1] * len(curr_input_ids))

        input_ids = create_padded_tensor(input_ids, self.tokenizer.padding_side, self.tokenizer.pad_token_id)
        attn_mask = create_padded_tensor(attn_mask, self.tokenizer.padding_side, 0)
        labels = create_padded_tensor(labels, self.tokenizer.padding_side, -100)

        if input_ids.size(1) > self.max_input_tokens:
            input_ids = input_ids[:, -self.max_input_tokens:]
            attn_mask = attn_mask[:, -self.max_input_tokens:]
            labels = labels[:, -self.max_input_tokens:]

        # To match the logic of transformers during generation
        position_ids = attn_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attn_mask.long() == 0, 1)

        return {
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'labels': labels,
            'targets': targets,
            'datasets': datasets,
            'position_ids': position_ids,
            'task_ids': task_ids,
        }
