import os
import argparse
import json
from src.model import AlpacaModel


def get_best_lr(model_root_dir: str, seed: int) -> str:
    best_lr, min_loss = None, float('inf')
    for lr in os.listdir(model_root_dir):
        model_dir = os.path.join(model_root_dir, lr, f'{seed}')
        with open(os.path.join(model_dir, 'validation.json')) as f:
            val_loss = json.load(f)['avg_loss']

        if val_loss < min_loss:
            min_loss = val_loss
            best_lr = lr

    return best_lr


def save_model(model_root_dir: str, save_prefix: str, save_adapters_only: bool, seed: int) -> None:
    best_lr = get_best_lr(model_root_dir, seed)
    model_dir = os.path.join(model_root_dir, best_lr, f'{seed}')
    print(f'Processing {model_dir}')

    peft_method = os.path.basename(model_root_dir)

    if save_adapters_only:
        hf_save_path = os.path.join('output_hf', f'{save_prefix}_peft_alpaca_{peft_method}_{best_lr}')
    else:
        hf_save_path = os.path.join('output_hf', f'{save_prefix}_alpaca_{peft_method}_{best_lr}')
    assert not os.path.exists(hf_save_path), f'{hf_save_path} already exists'

    # Load model
    ckpt_path = os.path.join(model_dir, 'saved_models', 'last.ckpt')
    model = AlpacaModel.load_from_checkpoint(ckpt_path, strict=False)

    # Save tokenizer
    model.tokenizer.save_pretrained(hf_save_path)

    # Save model
    if save_adapters_only:
        model.model.save_pretrained(hf_save_path)
    else:
        print('Merging adapter layers')
        merged_model = model.model.merge_and_unload(safe_merge=True)
        print(f'Saving to {hf_save_path}')
        merged_model.save_pretrained(hf_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script to save HF models for MT-Bench')
    parser.add_argument('--model_root_dir', type=str, required=True)
    parser.add_argument('--save_prefix', type=str, required=True)
    parser.add_argument('--save_adapters_only', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    save_model(args.model_root_dir, args.save_prefix, args.save_adapters_only, args.seed)
