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
from src.model import GenerationModel
from src.data import EvalDataset, GSM8kDataset, HumanEvalDataset


logger = logging.getLogger(__name__)

COMMONSENSE_DATASETS = [
    'arc_challenge',
    'arc_easy',
    'boolq',
    'hellaswag',
    'openbookqa',
    'piqa',
    'social_i_qa',
    'winogrande',
]


@hydra.main(version_base=None, config_path='src/conf/generation', config_name='test')
def main(args : DictConfig):
    logger.info(f'Config: {args}')
    if args.seed is not None:
        L.seed_everything(args.seed)

    assert os.path.exists(args.saved_path), f'{args.saved_path} does not exist'

    val_file = os.path.join(args.saved_path, 'validation.json')
    assert args.skip_val_check or os.path.exists(val_file), f'Validation file does not exist. Is training complete?'

    results_filepath = Path(args.saved_path, f'{args.results_filename}.json.gz')
    results_filepath.touch(exist_ok=args.overwrite_results_file)

    # Load dataset
    if args.dataset.name in COMMONSENSE_DATASETS:
        test_dataset = EvalDataset(**args.dataset)
    elif args.dataset.name == 'gsm8k':
        test_dataset = GSM8kDataset(split='test')
    elif args.dataset.name == 'humaneval':
        test_dataset = HumanEvalDataset()
    else:
        raise NotImplementedError(f'Dataset: {args.dataset.name}')

    ckpt_path = os.path.join(args.saved_path, 'saved_models', f'{args.ckpt_name}.ckpt')
    model = GenerationModel.load_from_checkpoint(ckpt_path, strict=False, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)

    # Trainer
    trainer = L.Trainer(
        accelerator='auto',
        devices=1,
        deterministic='warn',
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
            'model_outputs': model.model_outputs,
        }, f, indent=4)

    # Write to a file for humaneval
    if args.dataset.name == 'humaneval':
        output_filepath = os.path.join(args.saved_path, f'{args.results_filename}_output.jsonl')
        assert not os.path.exists(output_filepath), f'{output_filepath} already exists'
        with open(output_filepath, 'wb') as f:
            for output in model.model_outputs:
                output = {
                    'task_id': output['task_id'],
                    'completion': output['pred_answer'],
                }
                f.write((json.dumps(output) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
