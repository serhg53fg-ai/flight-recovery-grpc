"""Explicit recovery schema initialization; application startup never runs DDL."""
import argparse

from apps.flight.storage.factory import storage_environment, mysql_config
from apps.flight.recovery.repository import initialize_schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['initialize'])
    parser.parse_args()
    config = storage_environment()
    if config['STORAGE_BACKEND'] != 'mysql':
        parser.error('恢复任务初始化必须配置FLIGHT_STORAGE_BACKEND=mysql')
    initialize_schema(mysql_config(config))
    print('Recovery schema initialized (v1)')


if __name__ == '__main__':
    main()
