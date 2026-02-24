import os
import json
import hydra
import logging
import gzip
import lightning as L
from functools import partial
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from src.model import ImageClassificationModel
from src.data import Cifar100Dataset, Food101Dataset, Flowers102Dataset, Resisc45Dataset


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='src/conf/image', config_name='test')
def main(args : DictConfig):
    logger.info(f'Config: {args}')
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)

    assert os.path.exists(args.saved_path), f'{args.saved_path} does not exist'

    val_file = os.path.join(args.saved_path, 'validation.json')
    assert args.skip_val_check or os.path.exists(val_file), f'Validation file does not exist. Is training complete?'

    results_filepath = Path(args.saved_path, f'{args.results_filename}.json.gz')
    results_filepath.touch(exist_ok=args.overwrite_results_file)

    # Load dataset
    if args.dataset.name == 'cifar100':
        test_dataset = Cifar100Dataset(
            split='test',
            val_split_frac=None,
            max_val_examples=None,
            data_seed=None,
        )
    elif args.dataset.name == 'food101':
        test_dataset = Food101Dataset(
            split='test',
            val_split_frac=None,
            max_val_examples=None,
            data_seed=None,
        )
    elif args.dataset.name == 'flowers102':
        test_dataset = Flowers102Dataset(split='test')
    elif args.dataset.name == 'resisc45':
        test_dataset = Resisc45Dataset(split='test')
    else:
        raise NotImplementedError(f'Dataset: {args.dataset.name}')

    ckpt_path = os.path.join(args.saved_path, 'saved_models', f'{args.ckpt_name}.ckpt')
    model = ImageClassificationModel.load_from_checkpoint(ckpt_path, strict=False)

    # Trainer
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        deterministic=True,
        logger=False,
    )

    # Dataloaders
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=partial(model.collate_fn, is_test=True),
    )

    # Test
    test_results = trainer.test(model, test_dataloader)
    assert len(test_results) == 1
    with gzip.open(results_filepath, 'wt') as f:
        json.dump({
            'args': OmegaConf.to_container(args),
            'results': test_results[0],
        }, f, indent=4)


if __name__ == '__main__':
    main()
