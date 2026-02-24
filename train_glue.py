import os
import json
import hydra
import wandb
import logging
import torch
import lightning as L
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from peft import estimate_gradient
from src.model import GlueClassificationModel
from src.data import GlueDataset
from src.data.glue import EVAL_METRICS


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
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        labels=batch['labels'],
        return_dict=True,
    )
    return output['loss']


class SaveFirstEpochCheckpoint(L.pytorch.callbacks.ModelCheckpoint):
    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0:
            super().on_epoch_end(trainer, pl_module)
        else:
            pass


@hydra.main(version_base=None, config_path='src/conf/glue', config_name='train')
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
    train_dataset = GlueDataset(task=args.task, split='train')
    val_dataset = GlueDataset(task=args.task, split='validation')

    # Load model
    assert args.model_type == 'bert', f'{args.model_type} is not supported'

    model = GlueClassificationModel(
        model_name_or_path=args.model_name_or_path,
        optim=args.optim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_input_len=args.max_input_len,
        training_mode=args.training_mode,
        num_labels=train_dataset.num_labels(),
        warmup_steps=args.warmup_steps,
        scheduler_type=args.lr_scheduler_type,
    )

    # Checkpointing
    checkpoint_dir = os.path.join(args.save_path, 'saved_models')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    eval_metric = EVAL_METRICS[args.task]
    best_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor=f'val/{eval_metric}',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=True,
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
        collate_fn=model.collate_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=model.collate_fn,
    )

    # Train
    # For models with `init_type=gradient`, estimate gradients
    if hasattr(args.training_mode, 'init_type') and args.training_mode.init_type == 'gradient':
        named_grads = estimate_gradient(
            model=model.model,
            dataloader=train_dataloader,
            compute_loss=compute_loss,
            use_gradient_checkpointing=False,
            num_batches=args.training_mode.num_batches,
        )
        model.set_named_gradients(named_grads)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save best validation accuracy
    with open(os.path.join(args.save_path, 'validation.json'), 'w') as f:
        json.dump({
            'n_examples': len(val_dataset),
            f'val/{eval_metric}': trainer.checkpoint_callback.best_model_score.item(),
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
