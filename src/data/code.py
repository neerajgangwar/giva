import re
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict
from src.utils.common import TEMPLATE_WITHOUT_INPUT, ALPACA_PREFIX_TEMPLATE


class CodeFeedbackDataset(Dataset):
    def __init__(self, split: str, seed: int, start_idx: int, end_idx: int):
        examples = load_dataset('m-a-p/CodeFeedback-Filtered-Instruction', split='train')
        examples.shuffle(seed=seed)
        if split == 'train':
            self.examples = examples.select(list(range(start_idx, end_idx)))
        elif split == 'val':
            self.examples = examples.select(list(range(start_idx, end_idx)))
        else:
            raise NotImplementedError(f'Split: {split}')


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        answer = example['answer']
        answer = "```".join(answer.split("```")[:2]) + "```"
        return {
            'input_text': TEMPLATE_WITHOUT_INPUT.format(instruction=example['query']),
            'output_text': answer,
            'dataset': 'code_feedback',
            'target': None,
        }


class HumanEvalDataset(Dataset):
    def __init__(self):
        self.examples = load_dataset('openai/openai_humaneval', split='test')


    def __len__(self):
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, str]:
        example = self.examples[index]
        return {
            'task_id': example['task_id'],
            'input_text': ALPACA_PREFIX_TEMPLATE.format(PROMPT=example['prompt']),
            'output_text': None,
            'dataset': 'humaneval',
            'target': None,
        }
