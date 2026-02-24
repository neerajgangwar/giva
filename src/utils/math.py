import re


METAMATH_RESPONSE_PATTERN = r'#### [-+]?\b\d+(?:\.\d+)?\b\n'
PRED_RESPONSE_PATTERN = r'####\s*([-+]?(?:\d+\.\d+|\.\d+|\d+))'


def extract_answer_metamath(answer: str):
    matches = re.findall(METAMATH_RESPONSE_PATTERN, answer)
    assert len(matches) == 1, (answer, len(matches))
    m = matches[0].replace('####', '').strip()
    assert f'#### {m}' in answer
    return float(m)


def extract_answer_gsm8k(answer: str):
    m = answer.split('####')[-1].strip()
    assert f'#### {m}' in answer
    m = m.replace(',', '')
    return float(m)


def extract_answer_pred(pred: str) -> float:
    match = re.search(PRED_RESPONSE_PATTERN, pred)
    if not match:
        return float('inf')

    result = match.group(1)
    try:
        return float(result.replace(",", ""))
    except:
        print(f"'{result}' can't be converted")
        return float('inf')
