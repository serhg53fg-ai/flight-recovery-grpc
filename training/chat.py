"""Validated chat prefix and explicit completion mask without heavy imports."""
from .prompt import parse_duration_output


def tokenize_example(example, tokenizer, max_length=8192):
    parse_duration_output(example['response'])
    messages = [{'role': 'user', 'content': example['prompt']}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    full = tokenizer.apply_chat_template(messages + [{'role': 'assistant', 'content': example['response']}],
             tokenize=False, add_generation_prompt=False, enable_thinking=False)
    if not full.startswith(prefix):
        raise ValueError('chat template generation prefix mismatch')
    prefix_ids = tokenizer(prefix, add_special_tokens=False)['input_ids']
    ids = tokenizer(full, add_special_tokens=False)['input_ids']
    if not prefix_ids or ids[:len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
        raise ValueError('chat token prefix mismatch or empty answer')
    if len(ids) > max_length:
        raise ValueError('training sequence exceeds length limit; truncation forbidden')
    return {'input_ids': ids, 'completion_mask': [0] * len(prefix_ids) + [1] * (len(ids) - len(prefix_ids))}
