import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Any
from torch.utils.data.dataset import random_split


class Cifar100Dataset(Dataset):
    def __init__(self, split: str, val_split_frac: float=0.1, max_val_examples: int=5000, data_seed: int=42):
        if split == 'test':
            self.examples = load_dataset('uoft-cs/cifar100', split=split)
            self.labels = self.examples.features['fine_label'].names
        else:
            examples = load_dataset('uoft-cs/cifar100', split='train')
            self.labels = examples.features['fine_label'].names
            total_examples = len(examples)
            num_val_examples = min(int(total_examples * val_split_frac), max_val_examples)
            num_train_examples = total_examples - num_val_examples
            train_examples, val_examples = random_split(
                examples,
                [num_train_examples, num_val_examples],
                generator=torch.Generator().manual_seed(data_seed),
            )
            if split == 'train':
                self.examples = train_examples
            elif split == 'val':
                self.examples = val_examples
            else:
                raise NotImplementedError(f'split: "{split}"')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        img = example['img']
        label = example['fine_label']
        return {
            'img': img,
            'label': label,
        }


    def num_labels(self) -> int:
        return len(self.labels)


class Food101Dataset(Dataset):
    def __init__(self, split: str, val_split_frac: float=0.1, max_val_examples: int=5000, data_seed: int=42):
        if split == 'test':
            self.examples = load_dataset('ethz/food101', split='validation')
            self.labels = self.examples.features['label'].names
        else:
            examples = load_dataset('ethz/food101', split='train')
            self.labels = examples.features['label'].names
            total_examples = len(examples)
            num_val_examples = min(int(total_examples * val_split_frac), max_val_examples)
            num_train_examples = total_examples - num_val_examples
            train_examples, val_examples = random_split(
                examples,
                [num_train_examples, num_val_examples],
                generator=torch.Generator().manual_seed(data_seed),
            )
            if split == 'train':
                self.examples = train_examples
            elif split == 'val':
                self.examples = val_examples
            else:
                raise NotImplementedError(f'split: "{split}"')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        img = example['image']
        label = example['label']
        return {
            'img': img,
            'label': label,
        }


    def num_labels(self) -> int:
        return len(self.labels)


class Flowers102Dataset(Dataset):
    def __init__(self, split: str):
        self.examples = load_dataset('dpdl-benchmark/oxford_flowers102', split=split)
        self.labels = self.examples.features['label'].names


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        img = example['image']
        label = example['label']
        return {
            'img': img,
            'label': label,
        }


    def num_labels(self) -> int:
        return len(self.labels)


class Resisc45Dataset(Dataset):
    def __init__(self, split: str):
        self.examples = load_dataset('timm/resisc45', split=split)
        self.labels = self.examples.features['label'].names


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            'img': example['image'],
            'label': example['label'],
        }


    def num_labels(self) -> int:
        return len(self.labels)


class ImageNetDataset(Dataset):
    def __init__(self, split: str, num_val_examples: int=5000, data_seed: int=42):
        if split == 'test':
            self.examples = load_dataset('ILSVRC/imagenet-1k', split='validation')
            labels = self.examples.features['label'].names
        else:
            examples = load_dataset('ILSVRC/imagenet-1k', split='train')
            labels = examples.features['label'].names
            train_examples, val_examples = random_split(
                examples,
                [len(examples) - num_val_examples, num_val_examples],
                generator=torch.Generator().manual_seed(data_seed),
            )
            if split == 'train':
                self.examples = train_examples
            elif split == 'validation':
                self.examples = val_examples
            else:
                NotImplementedError

        self.labels = [label.split(',')[0].strip() for label in labels]


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        return {
            'img': example['image'],
            'label': example['label'],
            'labels': self.labels,
        }


    def num_labels(self) -> int:
        return len(self.labels)
