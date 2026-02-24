from typing import List, Dict, Any
from transformers import PreTrainedTokenizer


PROMPT_TEMPLATES = {
    'cola': '*cls**sent_0*_This_is*mask*.*sep+*',
    'mrpc': '*cls**sent_0**mask*,*+sentl_1**sep+*',
    'qnli': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'rte': '*cls**sent-_0*?*mask*,*+sentl_1**sep+*',
    'sst2': '*cls**sent_0*_It_was*mask*.*sep+*',
}
LABEL_MAPPING = {
    'cola': {'unacceptable': 'incorrect', 'acceptable': 'correct'},
    'mrpc': {'not_equivalent': 'No', 'equivalent': 'Yes'},
    'qnli': {'not_entailment': 'No', 'entailment': 'Yes'},
    'rte': {'not_entailment': 'No', 'entailment': 'Yes'},
    'sst2': {'negative': 'terrible', 'positive': 'great'},
}


def get_input_text_list(task: str, example: Dict[str, Any]) -> List[str]:
    if task in ('rte', 'mrpc'):
        return [example['sentence1'], example['sentence2']]
    elif task in ('sst2', 'cola'):
        return [example['sentence']]
    elif task in ('qnli'):
        return [example['question'], example['sentence']]
    else:
        raise NotImplementedError(f'For {task}')


def tokenize_label(tokenizer: PreTrainedTokenizer, label: str) -> int:
    target = tokenizer(' ' + label, add_special_tokens=False)['input_ids']
    assert len(target) == 1
    return target[0]



def tokenize_multipart_input(
    example: Dict[str, Any],
    max_length: int,
    truncate_head: bool,
    tokenizer: PreTrainedTokenizer,
    task: str,
    label: str,
):
    template = PROMPT_TEMPLATES[task]
    input_text_list = get_input_text_list(task, example)

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    input_ids = []
    attention_mask = []
    token_type_ids = [] # Only for BERT
    mask_pos = None # Position of the mask token

    """
    Concatenate all sentences and prompts based on the provided template.
    Template example: '*cls*It was*mask*.*sent_0**<sep>*label_0:*sent_1**<sep>**label_1*:*sent_2**<sep>*'
    *xx* represent variables:
        *cls*: cls_token
        *mask*: mask_token
        *sep*: sep_token
        *sep+*: sep_token, also means +1 for segment id
        *sent_i*: sentence i (input_text_list[i])
        *sent-_i*: same as above, but delete the last token
        *sentl_i*: same as above, but use lower case for the first word
        *sentl-_i*: same as above, but use lower case for the first word and delete the last token
        *+sent_i*: same as above, but add a space before the sentence
        *+sentl_i*: same as above, but add a space before the sentence and use lower case for the first word
        *label_i*: label_word_list[i]
        *label_x*: label depends on the example id (support_labels needed). this is only used in GPT-3's in-context learning

    Use "_" to replace space.
    PAY ATTENTION TO SPACE!! DO NOT leave space before variables, for this will lead to extra space token.
    """
    assert template is not None

    special_token_mapping = {
        'cls': tokenizer.cls_token_id,
        'mask': tokenizer.mask_token_id,
        'sep': tokenizer.sep_token_id,
        'sep+': tokenizer.sep_token_id,
    }
    template_list = template.split('*') # Get variable list in the template
    segment_id = 0 # Current segment id. Segment id +1 if encountering sep+.

    for part_id, part in enumerate(template_list):
        new_tokens = []
        segment_plus_1_flag = False
        if part in special_token_mapping:
            if part == 'cls' and 'T5' in type(tokenizer).__name__:
                # T5 does not have cls token
                continue
            new_tokens.append(special_token_mapping[part])
            if part == 'sep+':
                segment_plus_1_flag = True
        elif part[:6] == 'label_':
            raise NotImplementedError
        elif part[:7] == 'labelx_':
            raise NotImplementedError
        elif part[:5] == 'sent_':
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id])
        elif part[:6] == '+sent_':
            # Add space
            sent_id = int(part.split('_')[1])
            new_tokens += enc(' ' + input_text_list[sent_id])
        elif part[:6] == 'sent-_':
            # Delete the last token
            sent_id = int(part.split('_')[1])
            new_tokens += enc(input_text_list[sent_id][:-1])
        elif part[:6] == 'sentl_':
            # Lower case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentl_':
            # Lower case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(' ' + text)
        elif part[:7] == 'sentl-_':
            # Lower case the first token and discard the last token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].lower() + text[1:]
            new_tokens += enc(text[:-1])
        elif part[:6] == 'sentu_':
            # Upper case the first token
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(text)
        elif part[:7] == '+sentu_':
            # Upper case the first token and add space
            sent_id = int(part.split('_')[1])
            text = input_text_list[sent_id]
            text = text[:1].upper() + text[1:]
            new_tokens += enc(' ' + text)
        else:
            # Just natural language prompt
            part = part.replace('_', ' ')
            # handle special case when T5 tokenizer might add an extra space
            if len(part) == 1:
                new_tokens.append(tokenizer.convert_tokens_to_ids(part))
            else:
                new_tokens += enc(part)

        if part[:4] == 'sent' or part[1:5] == 'sent':
            # If this part is the sentence, limit the sentence length
            sent_id = int(part.split('_')[1])

        input_ids += new_tokens
        attention_mask += [1 for i in range(len(new_tokens))]
        token_type_ids += [segment_id for i in range(len(new_tokens))]

        if segment_plus_1_flag:
            segment_id += 1

    # Truncate
    if len(input_ids) > max_length:
        if truncate_head:
            input_ids = input_ids[-max_length:]
            attention_mask = attention_mask[-max_length:]
            token_type_ids = token_type_ids[-max_length:]
        else:
            # Default is to truncate the tail
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

    # Find mask token
    if tokenizer.mask_token_id in input_ids:
        mask_pos = input_ids.index(tokenizer.mask_token_id)
    else:
        mask_pos = -1

    result = {'input_ids': input_ids, 'attention_mask': attention_mask}
    if 'BERT' in type(tokenizer).__name__:
        # Only provide token type ids for BERT
        result['token_type_ids'] = token_type_ids

    result['mask_pos'] = mask_pos

    target = tokenize_label(tokenizer, LABEL_MAPPING[task][label])
    result['target'] = target

    result['labels'] = [-100] * len(input_ids)
    if mask_pos != -1:
        result['labels'][mask_pos] = target

    return result
