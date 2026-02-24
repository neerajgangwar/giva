import os
import json
import hydra
import logging
import torch
import lightning as L
from tqdm import tqdm
from pathlib import Path
from typing import Dict
from omegaconf import DictConfig
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.model import AlpacaModel


logger = logging.getLogger(__name__)


def tokenize(tokenizer: PreTrainedTokenizer, example: Dict[str, str]) -> Dict[str, torch.Tensor]:
    input_text = f'{example["input_text"]} '
    input_len = len(tokenizer(input_text)['input_ids'])

    text = f'{input_text}{example["output_text"]}'
    tokenized = tokenizer(text, return_tensors='pt')
    labels = tokenized['input_ids'].clone()
    labels[:, :input_len] = -100
    return {
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': labels,
    }


@torch.no_grad()
def compute_loss(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, example: Dict[str, str]):
    model_input = tokenize(tokenizer, example)
    model_input = {k: v.to(model.device) for k, v in model_input.items()}
    output = model(**model_input, return_dict=True)
    return output['loss'].item()


@hydra.main(version_base=None, config_path='src/conf/alpaca', config_name='test')
def main(args : DictConfig):
    logger.info(f'Config: {args}')
    if args.seed is not None:
        L.seed_everything(args.seed)

    val_filepath = os.path.join(args.saved_path, f'{args.results_filename}.json')
    Path(val_filepath).touch(exist_ok=args.overwrite_results_file)

    ckpt_path = os.path.join(args.saved_path, 'saved_models', f'{args.ckpt_name}.ckpt')
    assert os.path.exists(ckpt_path), f'{ckpt_path} does not exist'

    model = AlpacaModel.load_from_checkpoint(ckpt_path, strict=False)
    model = model.to('cuda')
    model.eval()

    with open(args.dataset_file) as f:
        val_examples = json.load(f)

    total_loss, n_examples = 0, 0
    for example in tqdm(val_examples):
        loss = compute_loss(model.model, model.tokenizer, example)
        total_loss += loss
        n_examples += 1

    with open(val_filepath, 'w') as f:
        json.dump({
            'n_examples': n_examples,
            'total_loss': total_loss,
            'avg_loss': total_loss / n_examples,
        }, f, indent=4)


if __name__ == '__main__':
    main()

