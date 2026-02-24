from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, Union


EVAL_METRICS = {
    'sst2': 'accuracy',
    'mrpc': 'accuracy',
    'qnli': 'accuracy',
    'rte': 'accuracy',
    'cola': 'matthews_correlation',
    'stsb': 'pearsonr',
}
LABEL_NAME_MAP = {
    'sst2': {0: 'negative', 1: 'positive', -1: 'other'},
    'mrpc': {0: 'different', 1: 'equivalent', -1: 'other'},
    'qnli': {0: 'entailment', 1: 'not_entailment', -1: 'other'},
    'rte': {0: 'entailment', 1: 'not_entailment', -1: 'other'},
    'cola': {0: 'unacceptable', 1: 'acceptable', -1: 'other'},
}

class GlueDataset(Dataset):
    def __init__(self, task: str, split: str):
        super(GlueDataset, self).__init__()
        self.task = task
        self.split = split
        self.examples = load_dataset('glue', self.task, split=self.split)
        if task == 'stsb':
            self.label_names = None
        else:
            self.label_names = self.examples.features['label'].names


    def __len__(self) -> int:
        return len(self.examples)


    def __getitem__(self, index: int) -> Dict[str, Union[str, int]]:
        example = self.examples[index]
        if self.task in ['sst2', 'cola']:
            return {
                'sentence': example['sentence'],
                'label': example['label'],
                'label_name': self.label_names[example['label']],
                'task': self.task,
            }
        elif self.task == 'qnli':
            return {
                'question': example['question'],
                'sentence': example['sentence'],
                'label': example['label'],
                'label_name': self.label_names[example['label']],
                'task': self.task,
            }
        elif self.task in ['mrpc', 'rte']:
            return {
                'sentence1': example['sentence1'],
                'sentence2': example['sentence2'],
                'label': example['label'],
                'label_name': self.label_names[example['label']],
                'task': self.task,
            }
        elif self.task in ['stsb']:
            return {
                'sentence1': example['sentence1'],
                'sentence2': example['sentence2'],
                'label': example['label'],
                'label_name': str([example['label']]),
                'task': self.task,
            }
        else:
            NotImplementedError(f'Task: {self.task}')


    def num_labels(self):
        if self.task == 'stsb':
            return 1
        else:
            return len(self.label_names)
