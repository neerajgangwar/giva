import os
import json
import hydra
import wandb
import logging
import lightning as L
from functools import partial
from typing import Dict, Any
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from peft import estimate_gradient
from src.model import GenerationModel
from src.data import CommonsenseReasoningDataset, MetaMathDataset, CodeFeedbackDataset


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


@hydra.main(version_base=None, config_path='src/conf/generation', config_name='train')
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
    if args.dataset.name == 'commonsense':
        train_dataset = CommonsenseReasoningDataset(filepath=args.dataset.filepath, split='train')
        val_dataset = CommonsenseReasoningDataset(filepath=args.dataset.filepath, split='val')
    elif args.dataset.name == 'metamath':
        train_dataset = MetaMathDataset(
            split='train',
            seed=args.dataset.data_seed,
            start_idx=args.dataset.train.start_idx,
            end_idx=args.dataset.train.end_idx,
            filter_gsm=args.dataset.filter_gsm,
        )
        val_dataset = MetaMathDataset(
            split='val',
            seed=args.dataset.data_seed,
            start_idx=args.dataset.val.start_idx,
            end_idx=args.dataset.val.end_idx,
            filter_gsm=args.dataset.filter_gsm,
        )
    elif args.dataset.name == 'code_feedback':
        train_dataset = CodeFeedbackDataset(
            split='train',
            seed=args.dataset.data_seed,
            start_idx=args.dataset.train.start_idx,
            end_idx=args.dataset.train.end_idx,
        )
        val_dataset = CodeFeedbackDataset(
            split='val',
            seed=args.dataset.data_seed,
            start_idx=args.dataset.val.start_idx,
            end_idx=args.dataset.val.end_idx,
        )
    else:
        raise NotImplementedError(f'For dataset {args.dataset.name}')

    # Load model
    model = GenerationModel(
        model_name_or_path=args.model_name_or_path,
        optim=args.optim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        training_mode=args.training_mode,
        warmup_steps=args.warmup_steps,
        max_new_tokens=args.max_gen_tokens,
        scheduler_type=args.lr_scheduler_type,
        train_on_inputs=args.train_on_inputs,
        max_input_tokens=args.max_input_tokens,
    )

    # Checkpointing
    checkpoint_dir = os.path.join(args.save_path, 'saved_models')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    best_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best',
        monitor='val/loss',
        mode='min',
        save_last=False,
        verbose=True,
        save_weights_only=True,
    )
    last_checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='last',
        save_on_train_epoch_end=True,
        every_n_epochs=1 if args.save_last_ckpt else (args.num_epochs + 1),
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
            L.pytorch.callbacks.LearningRateMonitor(),
            timer,
            best_checkpoint_callback,
            last_checkpoint_callback,
        ],
        val_check_interval=args.val_check_interval,
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
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=partial(model.collate_fn, is_test=False),
    )

    # Train
    # For models with `init_type=gradient`, estimate gradients
    if hasattr(args.training_mode, 'init_type') and args.training_mode.init_type == 'gradient':
        named_grads = estimate_gradient(
            model=model.model,
            dataloader=train_dataloader,
            compute_loss=compute_loss,
            num_batches=args.training_mode.num_batches,
            mixed_precision='no',
        )
        model.set_named_gradients(named_grads)

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # Save best validation accuracy
    with open(os.path.join(args.save_path, 'validation.json'), 'w') as f:
        json.dump({
            'n_examples': len(val_dataset),
            'val/loss': trainer.checkpoint_callback.best_model_score.item(),
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
