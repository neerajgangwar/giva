import re
from typing import Dict
from .common import TEMPLATE_WITH_INPUT, TEMPLATE_WITHOUT_INPUT


def generate_prompt(data_point: Dict[str, str]) -> str:
    if data_point['input']:
        return TEMPLATE_WITH_INPUT.format(instruction=data_point['instruction'], input=data_point['input'])
    else:
        return TEMPLATE_WITHOUT_INPUT.format(instruction=data_point['instruction'])


def extract_answer(dataset: str, sentence: str) -> str:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'arc_challenge', 'arc_easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    else:
        raise NotImplementedError(f'dataset: {dataset}')
