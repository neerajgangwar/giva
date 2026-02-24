import torch
from typing import Dict, Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from torch.nn.utils.rnn import pad_sequence
from peft import prepare_model_for_kbit_training
from omegaconf import DictConfig
from src.utils.common import create_padded_tensor
from .base import BaseModel


TEMPLATE_WITH_INPUT = '''### Instruction:
{instruction}

### Input:
{input}

### Response:
'''
TEMPLATE_WITHOUT_INPUT = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''


class AlpacaModel(BaseModel):
    def __init__(
        self,
        model_name_or_path: str,
        optim: str,
        lr: float,
        weight_decay: float,
        training_mode: DictConfig,
        scheduler_type: str,
        warmup_steps: float,
        generation_config: Dict[str, Any]=None,
        max_input_tokens: int=2048,
        quantize: bool=True,
    ):
        self.generation_config = generation_config
        self.quantize = quantize
        super(AlpacaModel, self).__init__(
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


    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, padding_side='right', legacy=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        if not tokenizer.bos_token:
            tokenizer.bos_token = tokenizer.eos_token
        return tokenizer


    def get_model(self):
        if self.quantize:
            raise NotImplementedError
        else:
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, torch_dtype=torch.bfloat16)
        return model


    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        batch_size = batch['input_ids'].size(0)
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            return_dict=True,
        )
        loss = output['loss']
        self.log('train/loss', loss, batch_size=batch_size, on_step=True, on_epoch=False, sync_dist=True, prog_bar=True)
        return loss


    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> float:
        raise NotImplementedError


    def collate_fn(self, batch: Any, is_test: bool=True) -> Dict[str, Any]:
        def _format_input(example: Dict[str, Any]) -> str:
            if example['input'].strip():
                input_text = TEMPLATE_WITH_INPUT.format(instruction=example['instruction'], input=example['input'])
            else:
                input_text = TEMPLATE_WITHOUT_INPUT.format(instruction=example['instruction'])
            return input_text

        # Extract elements
        sources = [f'{self.tokenizer.bos_token}{_format_input(example)}' for example in batch]
        full = [
            f'{self.tokenizer.bos_token}{_format_input(example)}{example["output"]}{self.tokenizer.eos_token}'
            for example in batch
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.max_input_tokens,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_full_text = self.tokenizer(
            full,
            max_length=self.max_input_tokens,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_full in zip(tokenized_sources_with_prompt['input_ids'], tokenized_full_text['input_ids']):
            if not is_test:
                input_ids.append(torch.tensor(tokenized_full))
                labels.append(torch.tensor([-100 for _ in range(len(tokenized_source))] + tokenized_full[len(tokenized_source):]))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=-100)
            if not is_test
            else None
        )
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict
