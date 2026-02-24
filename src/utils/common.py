import torch
from typing import List, Any


TEMPLATE_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

TEMPLATE_WITHOUT_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

ALPACA_PREFIX_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Complete the following Python code:
Notes: respond with the entire complete function definition
do not add any comments, be as concise in your code as possible
use only built-in libraries, assume no additional imports other than those provided (if any)
use `    ` (4 spaces) for each level of indentation

code:
```python
{PROMPT}
```

### Response:
```python
"""


def create_padded_tensor(ll: List[List[Any]], padding_side: str, fill_val: Any, torch_dtype: torch.dtype=torch.long) -> torch.Tensor:
    batch_size = len(ll)
    lens = [len(l) for l in ll]

    out_tensor = torch.empty(batch_size, max(lens), dtype=torch_dtype).fill_(fill_val)

    for batch_idx in range(len(ll)):
        l = lens[batch_idx]
        if padding_side == 'left':
            out_tensor[batch_idx, -l:] = torch.tensor(ll[batch_idx], dtype=torch_dtype)
        else:
            out_tensor[batch_idx, :l] = torch.tensor(ll[batch_idx], dtype=torch_dtype)

    return out_tensor
