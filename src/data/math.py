from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict
from src.utils.math import extract_answer_gsm8k, extract_answer_metamath


class MetaMathDataset(Dataset):
    def __init__(self, split: str, seed: int, start_idx: int, end_idx: int, filter_gsm: bool):
        examples = load_dataset('meta-math/MetaMathQA', split='train')
        examples.shuffle(seed=seed)
        examples = [e for e in examples if not filter_gsm or 'GSM' in e['type']]
        if split == 'train':
            self.examples = examples[start_idx:end_idx]
        elif split == 'val':
            self.examples = examples[start_idx:end_idx]
        else:
            raise NotImplementedError(f'Split: {split}')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        if 'GSM' in example['type']:
            target = extract_answer_metamath(example['response'])
        else:
            target = None
        return {
            'input_text': f'Q: {example["query"]}\nA: ',
            'output_text': example['response'],
            'target': target,
            'dataset': 'metamath',
        }


class GSM8kDataset(Dataset):
    def __init__(self, split: str):
        self.examples = load_dataset('gsm8k', 'main', split=split)


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        return {
            'input_text': f'Q: {example["question"]}\nA: ',
            'output_text': example['answer'],
            'target': extract_answer_gsm8k(example['answer']),
            'dataset': 'gsm8k',
        }
