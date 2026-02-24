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
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from peft import estimate_gradient
from src.model import ImageClassificationModel
from src.data import Cifar100Dataset, Food101Dataset, Flowers102Dataset, Resisc45Dataset, ImageNetDataset


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


def compute_loss(model: PreTrainedModel, batch: Dict[str, Any]) -> float:
    output = model(
        pixel_values=batch['pixel_values'],
        labels=batch['labels'],
        return_dict=True,
    )
    return output['loss']


@hydra.main(version_base=None, config_path='src/conf/image', config_name='train')
def main(args : DictConfig):
    logger.info(f'Config: {args}')
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    assert args.save_path, f'{args.save_path} is empty'

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=args.overwrite_save_path)

    args_filepath = os.path.join(args.save_path, 'args.yaml')
    OmegaConf.save(args, args_filepath)

    # Load dataset
    if args.dataset.name == 'cifar100':
        train_dataset = Cifar100Dataset(
            split='train',
            val_split_frac=args.dataset.val_split_frac,
            max_val_examples=args.dataset.max_val_examples,
            data_seed=args.dataset.data_seed,
        )
        val_dataset = Cifar100Dataset(
            split='val',
            val_split_frac=args.dataset.val_split_frac,
            max_val_examples=args.dataset.max_val_examples,
            data_seed=args.dataset.data_seed,
        )
        num_labels = train_dataset.num_labels()
    elif args.dataset.name == 'food101':
        train_dataset = Food101Dataset(
            split='train',
            val_split_frac=args.dataset.val_split_frac,
            max_val_examples=args.dataset.max_val_examples,
            data_seed=args.dataset.data_seed,
        )
        val_dataset = Food101Dataset(
            split='val',
            val_split_frac=args.dataset.val_split_frac,
            max_val_examples=args.dataset.max_val_examples,
            data_seed=args.dataset.data_seed,
        )
        num_labels = train_dataset.num_labels()
    elif args.dataset.name == 'flowers102':
        train_dataset = Flowers102Dataset(split='train')
        val_dataset = Flowers102Dataset(split='validation')
        num_labels = train_dataset.num_labels()
    elif args.dataset.name == 'resisc45':
        train_dataset = Resisc45Dataset(split='train')
        val_dataset = Resisc45Dataset(split='validation')
        num_labels = train_dataset.num_labels()
    elif args.dataset.name == 'imagenet':
        train_dataset = ImageNetDataset(
            split='train',
            num_val_examples=args.dataset.num_val_examples,
            data_seed=args.dataset.data_seed,
        )
        val_dataset = ImageNetDataset(
            split='validation',
            num_val_examples=args.dataset.num_val_examples,
            data_seed=args.dataset.data_seed,
        )
        num_labels = train_dataset.num_labels()
    else:
        raise NotImplementedError(f'For dataset "{args.dataset.name}"')

    # Load model
    if args.training_mode in ('vera', 'randlora'):
        assert args.training_mode.config is not None
        # Taken from https://github.com/PaulAlbert31/RandLoRA/blob/371d82f4ef713fc4c35d6cb581d37eef558dbb31/GLUE/run_glue.py
        args.training_mode.config.projection_prng_key = int(torch.exp(torch.tensor(args.seed)) * 3.1415 * 1)

    model = ImageClassificationModel(
        model_name_or_path=args.model_name_or_path,
        optim=args.optim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        training_mode=args.training_mode,
        warmup_steps=args.warmup_steps,
        scheduler_type=args.lr_scheduler_type,
        num_labels=num_labels,
    )

    # Checkpointing
    checkpoint_dir = os.path.join(args.save_path, 'saved_models')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor=f'val/accuracy',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )

    # Time
    timer = L.pytorch.callbacks.Timer()

    # Trainer
    if model.model.config.model_type == 'dinov2' and args.training_mode.type == 'fft':
        deterministic = 'warn'
    else:
        deterministic = True
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val,
        deterministic=deterministic,
        logger=[load_csv_logger(args)],
        callbacks=[
            best_checkpoint_callback,
            L.pytorch.callbacks.LearningRateMonitor(),
            L.pytorch.callbacks.DeviceStatsMonitor(),
            timer,
        ],
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision=args.precision,
        check_val_every_n_epoch=1,
    )

    # Dataloaders
    assert args.train_batch_size % (trainer.num_devices * args.accumulate_grad_batches) == 0
    train_batch_size = args.train_batch_size // (trainer.num_devices * args.accumulate_grad_batches)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=partial(model.collate_fn, is_test=False),
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=partial(model.collate_fn, is_test=True),
        num_workers=8,
    )

    # Train
    # For models with `init_type=gradient`, estimate gradients
    if hasattr(args.training_mode, 'init_type') and args.training_mode.init_type == 'gradient':
        if model.model.config.model_type == 'dinov2':
            torch.use_deterministic_algorithms(True, warn_only=True)

        named_grads = estimate_gradient(
            model=model.model,
            dataloader=train_dataloader,
            compute_loss=compute_loss,
            use_gradient_checkpointing=False,
            num_batches=args.training_mode.num_batches,
        )
        model.set_named_gradients(named_grads)

        torch.use_deterministic_algorithms(True)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save best validation accuracy
    with open(os.path.join(args.save_path, 'validation.json'), 'w') as f:
        json.dump({
            'n_examples': len(val_dataset),
            f'val/accuracy': trainer.checkpoint_callback.best_model_score.item(),
        }, f, indent=4)

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
