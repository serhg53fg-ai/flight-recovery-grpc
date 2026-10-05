import json
import pytest


class Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        prefix = '<user>' + messages[0]['content'] + '</user><assistant>'
        return prefix if add_generation_prompt else prefix + messages[1]['content'] + '</assistant>'
    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': list(text.encode())}


def example():
    return {'prompt': 'flight data', 'response': json.dumps(dict(off_block_delay_min=0,
            taxi_out_min=10, airborne_min=100, taxi_in_min=5))}


def test_only_answer_tokens_receive_loss():
    from training.chat import tokenize_example
    tokenized = tokenize_example(example(), Tokenizer())
    prefix = Tokenizer().apply_chat_template([{'role': 'user', 'content': example()['prompt']}],
                                             add_generation_prompt=True)
    assert tokenized['completion_mask'][:len(prefix)] == [0] * len(prefix)
    assert all(tokenized['completion_mask'][len(prefix):])
    assert bytes(tokenized['input_ids'][len(prefix):]).decode().startswith(example()['response'])


def test_rejects_truncation_and_invalid_answer():
    from training.chat import tokenize_example
    with pytest.raises(ValueError, match='length'):
        tokenize_example(example(), Tokenizer(), max_length=5)
    with pytest.raises(ValueError):
        tokenize_example({**example(), 'response': '{}'}, Tokenizer())


def test_rejects_template_prefix_mismatch():
    from training.chat import tokenize_example
    class Broken(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            text = super().apply_chat_template(messages, **kwargs)
            return text if kwargs.get('add_generation_prompt') else 'different' + text
    with pytest.raises(ValueError, match='prefix'):
        tokenize_example(example(), Broken())
