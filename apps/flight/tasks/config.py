"""Explicit async deployment configuration; no credentials in capability output."""
import math
import os

from .documents import bounded_integer, label
from apps.flight.domain.model_contract import validate_source


def durable_environment():
    enabled = os.environ.get('FLIGHT_DURABLE_ENABLED', '0')
    if enabled not in ('0', '1'):
        raise ValueError('FLIGHT_DURABLE_ENABLED必须为0或1')
    return dict(DURABLE_ENABLED=enabled == '1',
                DURABLE_MODEL_VERSION=os.environ.get('FLIGHT_DURABLE_MODEL_VERSION'),
                DURABLE_SOURCE=os.environ.get('FLIGHT_DURABLE_SOURCE'),
                DURABLE_PROMPT_VERSION=os.environ.get('FLIGHT_DURABLE_PROMPT_VERSION'),
                DURABLE_TTL_SECONDS=int(os.environ.get('FLIGHT_DURABLE_TTL_SECONDS', '300')),
                DURABLE_MAX_ATTEMPTS=int(os.environ.get('FLIGHT_DURABLE_MAX_ATTEMPTS', '3')),
                SSE_POLL_SECONDS=.5, SSE_HEARTBEAT_SECONDS=15, SSE_MAX_SECONDS=30)


def validate_durable_config(config):
    if type(config['DURABLE_ENABLED']) is not bool:
        raise ValueError('DURABLE_ENABLED必须为布尔值')
    if not config['DURABLE_ENABLED']:
        return
    if config.get('STORAGE_BACKEND') != 'mysql':
        raise ValueError('持久异步模式必须使用mysql')
    label(config.get('DURABLE_MODEL_VERSION'), '模型版本')
    label(config.get('DURABLE_PROMPT_VERSION'), 'Prompt版本')
    validate_source(config.get('DURABLE_SOURCE'))
    bounded_integer(config['DURABLE_TTL_SECONDS'], 1, 3600, '任务期限')
    bounded_integer(config['DURABLE_MAX_ATTEMPTS'], 1, 5, '领取次数')
    for name, minimum, maximum in [('SSE_POLL_SECONDS', .01, 5),
                                  ('SSE_HEARTBEAT_SECONDS', .05, 60), ('SSE_MAX_SECONDS', .1, 300)]:
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError('无效SSE时间配置')


def configured_repository(config):
    if not config['DURABLE_ENABLED']:
        return None
    from apps.flight.storage.factory import mysql_config
    from .repository import DurableRepository
    return DurableRepository(mysql_config(config))
