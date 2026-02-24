import os
import json
import hydra
import wandb
import logging
import torch
import lightning as L
from functools import partial
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from transformers import PreTrainedModel
from torch.utils.data import DataLoader
from peft import estimate_gradient
from src.model import AlpacaModel
from src.data import AlpacaDataset


logger = logging.getLogger(__name__)


def load_wandb_logger(args):
    # wandb
    wandbfile = os.path.join(args.save_path, 'wandb.yaml')
    wandbconfig = {
        'id': wandb.util.generate_id(),
        'name': args.run_name,
    }
    with open(wandbfile, 'w') as f:
        json.dump(wandbconfig, f, indent=4)

    return L.pytorch.loggers.WandbLogger(
        id = wandbconfig['id'],
        name = wandbconfig['name'],
        save_dir = args.save_path,
        project = args.project_name,
        log_model = False,
    )


def load_csv_logger(args):
    return L.pytorch.loggers.CSVLogger(
        save_dir=args.save_path,
        name='train_logs',
    )


def compute_loss(model: PreTrainedModel, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    output = model(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        return_dict=True,
    )
    return output['loss']


@hydra.main(version_base=None, config_path='src/conf/alpaca', config_name='train')
def main(args : DictConfig):
    logger.info(f'Config: {args}')
    if args.seed is not None:
        L.seed_everything(args.seed)

    assert args.save_path, f'{args.save_path} is empty'

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=args.overwrite_save_path)

    args_filepath = os.path.join(args.save_path, 'args.yaml')
    OmegaConf.save(args, args_filepath)

    # Load dataset
    train_dataset = AlpacaDataset(split='train')

    # Load model
    model = AlpacaModel(
        model_name_or_path=args.model_name_or_path,
        optim=args.optim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        training_mode=args.training_mode,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.lr_scheduler_type,
        max_input_tokens=args.max_input_tokens,
        quantize=args.quantize,
    )

    # Checkpointing
    checkpoint_dir = os.path.join(args.save_path, 'saved_models')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='last',
        verbose=True,
        save_weights_only=True,
        every_n_epochs=1,
        save_top_k=1,
    )

    # Time
    timer = L.pytorch.callbacks.Timer()

    # Trainer
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=True,
        logger=[load_csv_logger(args)] + ([load_wandb_logger(args)] if args.enable_wandb else []),
        callbacks=[
            best_checkpoint_callback,
            L.pytorch.callbacks.LearningRateMonitor(),
            timer,
        ],
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Dataloaders
    assert args.train_batch_size % (trainer.num_devices * args.accumulate_grad_batches) == 0
    train_batch_size = args.train_batch_size // (trainer.num_devices * args.accumulate_grad_batches)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(model.collate_fn, is_test=False),
    )

    # Train
    # For models with `init_type=gradient`, estimate gradients
    if hasattr(args.training_mode, 'init_type') and args.training_mode.init_type == 'gradient':
        named_grads = estimate_gradient(
            model=model.model,
            dataloader=train_dataloader,
            num_batches=args.training_mode.num_batches,
            compute_loss=compute_loss,
            mixed_precision='no',
            use_gradient_checkpointing=False,
        )
        model.set_named_gradients(named_grads)

    trainer.fit(model, train_dataloaders=train_dataloader)

    # Save parameter stats
    total_params = sum(p.numel() for n, p in model.named_parameters())
    trainable_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    peft_params = sum(
        p.numel() for n, p in model.named_parameters()
        if args.training_mode.type in n
    )
    peft_trainable_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and args.training_mode.type in n
    )
    with open(os.path.join(args.save_path, 'parameter_stats.json'), 'w') as f:
        json.dump({
            'total_params': total_params,
            'trainable_params': trainable_params,
            'peft_params': peft_params,
            'peft_trainable_params': peft_trainable_params,
        }, f, indent=4)


    with open(os.path.join(args.save_path, 'time_stats.json'), 'w') as f:
        json.dump({
            'training_time': timer.time_elapsed('train'),
            'validation_time': timer.time_elapsed('validate'),
        }, f, indent=4)


if __name__ == '__main__':
    main()
