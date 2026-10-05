"""Shared prediction-source contract for durable jobs and recovery."""

PREDICTION_SOURCES = ('TEST', 'LLM', 'BASELINE')


def validate_source(value: str) -> str:
    if value not in PREDICTION_SOURCES or not isinstance(value, str):
        raise ValueError('无效预测来源')
    return value
