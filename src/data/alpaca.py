from torch.utils.data import Dataset
from typing import Dict
from datasets import load_dataset


class AlpacaDataset(Dataset):
    def __init__(self, split: str):
        self.examples = load_dataset('yahma/alpaca-cleaned', split='train')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        return {
            'instruction': example['instruction'],
            'input': example['input'],
            'output': example['output'],
        }
