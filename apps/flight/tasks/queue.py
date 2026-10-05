"""Bounded Redis 6 Streams notifications; MySQL remains authoritative."""
import redis

from apps.flight.storage.documents import validate_job_id
from .documents import bounded_integer, label


class QueueError(RuntimeError):
    pass


PUBLISH = """
if redis.call('XLEN',KEYS[1]) >= tonumber(ARGV[1]) then
    return redis.error_reply('QUEUE_CAPACITY')
end
return redis.call('XADD',KEYS[1],'*','message_id',ARGV[2],'job_id',ARGV[3],'flight_id',ARGV[4])
"""
ACK = """
local n = redis.call('XACK',KEYS[1],ARGV[1],ARGV[2])
if n > 0 then redis.call('XDEL',KEYS[1],ARGV[2]) end
return n
"""


def payload(value):
    if not isinstance(value, dict) or set(value) != {'message_id', 'job_id', 'flight_id'}:
        raise ValueError('无效队列通知')
    for item in value.values():
        validate_job_id(item)
    return value


class StreamQueue:
    def __init__(self, host='127.0.0.1', port=6379, password=None, unix_socket=None,
                 stream='flight:predictions', group='prediction-executors', capacity=2000):
        label(host, 'Redis主机')
        bounded_integer(port, 1, 65535, 'Redis端口')
        bounded_integer(capacity, 1, 100000, '队列容量')
        label(stream, 'Stream名称')
        label(group, '消费者组')
        if unix_socket is not None:
            from pathlib import Path
            if not isinstance(unix_socket, str) or not Path(unix_socket).is_absolute():
                raise ValueError('Redis socket必须为绝对路径')
        self.stream, self.group, self.capacity = stream, group, capacity
        settings = dict(password=password, decode_responses=True, socket_timeout=5,
                        socket_connect_timeout=5, retry_on_timeout=False)
        if unix_socket:
            settings['unix_socket_path'] = unix_socket
        else:
            settings.update(host=host, port=port)
        self.client = redis.Redis(**settings)

    def _ensure_group(self):
        try:
            self.client.xgroup_create(self.stream, self.group, id='0-0', mkstream=True)
        except redis.ResponseError as exc:
            if not str(exc).startswith('BUSYGROUP'):
                raise

    def publish(self, value):
        payload(value)
        bounded_integer(self.capacity, 1, 100000, '队列容量')
        try:
            return self.client.eval(PUBLISH, 1, self.stream, self.capacity,
                                    value['message_id'], value['job_id'], value['flight_id'])
        except redis.RedisError:
            raise QueueError('队列不可用或达到容量上限') from None

    def receive(self, consumer, idle_ms=30000):
        label(consumer, '消费者')
        bounded_integer(idle_ms, 1, 3600000, '消息回收等待')
        try:
            self._ensure_group()
            pending = self.client.xpending_range(self.stream, self.group, '-', '+', 100)
            for item in pending:
                if item['time_since_delivered'] >= idle_ms:
                    messages = self.client.xclaim(self.stream, self.group, consumer, idle_ms, [item['message_id']])
                    if messages:
                        message_id, value = messages[0]
                        return message_id, payload(value)
            streams = self.client.xreadgroup(self.group, consumer, {self.stream: '>'}, count=1)
            if streams:
                message_id, value = streams[0][1][0]
                return message_id, payload(value)
            return None
        except (redis.RedisError, ValueError):
            raise QueueError('队列读取失败或通知格式不兼容') from None

    def ack(self, message_id):
        label(message_id, '队列消息ID')
        try:
            self.client.eval(ACK, 1, self.stream, self.group, message_id)
        except redis.RedisError:
            raise QueueError('队列确认失败；数据库结果保持不变') from None

    def close(self):
        self.client.close()
