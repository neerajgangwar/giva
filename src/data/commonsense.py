from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict
from src.utils.commonsense import generate_prompt


class CommonsenseReasoningDataset(Dataset):
    def __init__(self, filepath: str, split: str, val_set_size: int=120):
        examples = load_dataset('json', data_files=filepath)
        train_val = examples['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        if split == 'train':
            self.examples = train_val['train']
        elif split == 'val':
            self.examples = train_val['test']
        else:
            raise NotImplementedError(f'split: {split}')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        return {
            'input_text': generate_prompt(example),
            'output_text': example['output'],
            'target': example['answer'],
            'dataset': 'commonsense',
        }


class EvalDataset(Dataset):
    def __init__(self, filepath: str, name: str):
        self.name = name
        self.examples = load_dataset('json', data_files=filepath)['train']


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        input_text = generate_prompt({**example, 'output': ''})
        return {
            'input_text': input_text,
            'output_text': input_text,
            'target': example['answer'],
            'dataset': self.name,
        }
