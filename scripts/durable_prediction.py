"""Operational CLI for phase B; no HTTP or model loading in this process."""
import argparse
import json
import os
import time

from apps.flight.domain.model_contract import PREDICTION_SOURCES

from apps.flight.storage.factory import mysql_config, storage_environment
from apps.flight.storage.mysql_jobs import initialize_schema
from apps.flight.storage.repository import StorageError
from apps.flight.tasks.repository import DurableRepository, initialize_durable
from apps.flight.tasks.queue import QueueError, StreamQueue


def queue_from_environment():
    return StreamQueue(host=os.environ.get('FLIGHT_REDIS_HOST', '127.0.0.1'),
                       port=int(os.environ.get('FLIGHT_REDIS_PORT', '6379')),
                       password=os.environ.get('FLIGHT_REDIS_PASSWORD'),
                       unix_socket=os.environ.get('FLIGHT_REDIS_UNIX_SOCKET'),
                       stream=os.environ.get('FLIGHT_REDIS_STREAM', 'flight:predictions'),
                       group=os.environ.get('FLIGHT_REDIS_GROUP', 'prediction-executors'),
                       capacity=int(os.environ.get('FLIGHT_QUEUE_CAPACITY', '2000')))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Durable prediction CLI (MySQL / Redis / gRPC)')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('initialize')
    submit = commands.add_parser('submit')
    submit.add_argument('--input', required=True)
    submit.add_argument('--key', required=True)
    submit.add_argument('--timezone', default='Asia/Shanghai')
    submit.add_argument('--model-version', required=True)
    submit.add_argument('--source', choices=PREDICTION_SOURCES, required=True)
    submit.add_argument('--prompt-version', required=True)
    submit.add_argument('--ttl', type=int, default=300)
    submit.add_argument('--max-attempts', type=int, default=3)
    status = commands.add_parser('status')
    status.add_argument('job_id')
    status.add_argument('--include-results', action='store_true')
    commands.add_parser('publisher').add_argument('--once', action='store_true')
    worker = commands.add_parser('executor')
    worker.add_argument('--owner', required=True)
    worker.add_argument('--lease', type=int, default=30)
    worker.add_argument('--once', action='store_true')
    repair = commands.add_parser('reconcile')
    repair.add_argument('--republish', action='store_true', help='maintenance: republish all unfinished notifications')
    repair.add_argument('--once', action='store_true')
    args = parser.parse_args(argv)
    queue = client = None
    try:
        config = mysql_config(storage_environment())
        if args.command == 'initialize':
            initialize_schema(config)
            initialize_durable(config)
            print(json.dumps({'initialized': True, 'durable_schema_version': 1}))
            return 0
        repo = DurableRepository(config)
        if args.command == 'submit':
            with open(args.input, encoding='utf-8') as source:
                flights = json.load(source)
            job = repo.submit(flights, args.timezone, args.key, args.model_version, args.source,
                              args.prompt_version, args.ttl, args.max_attempts)
            print(json.dumps({'job_id': job['job_id'], 'status': job['status']}))
            return 0
        if args.command == 'status':
            job = repo.get(args.job_id)
            excluded = ('flights',) if args.include_results else ('flights', 'results')
            print(json.dumps({k: v for k, v in job.items() if k not in excluded}, ensure_ascii=False))
            return 0
        if args.command == 'publisher':
            from apps.flight.tasks.publisher import OutboxPublisher
            queue = queue_from_environment()
            action = OutboxPublisher(repo, queue).run_once
        elif args.command == 'executor':
            from apps.flight.clients.inference import InferenceClient
            from apps.flight.tasks.executor import DurableExecutor
            queue = queue_from_environment()
            client = InferenceClient(os.environ.get('FLIGHT_GATEWAY_ADDRESS', '127.0.0.1:50051'),
                                     timeout=os.environ.get('FLIGHT_RPC_TIMEOUT', '20'))
            action = DurableExecutor(repo, queue, client, args.owner, args.lease).run_once
        else:
            if args.republish and not args.once:
                raise ValueError('队列重建必须显式使用--once维护模式')
            action = lambda: repo.reconcile(republish=args.republish)
        while True:
            try:
                count = action()
                print(json.dumps({'role': args.command, 'processed': int(count)}), flush=True)
            except (StorageError, QueueError):
                print('持久执行依赖暂不可用；已提交数据保持在MySQL中', flush=True)
                if args.once:
                    return 2
            if args.once:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    except (StorageError, QueueError, ValueError, OSError, KeyError, TypeError):
        print('持久预测操作失败；请检查配置、输入、schema及任务状态')
        return 2
    finally:
        if client:
            client.close()
        if queue:
            queue.close()


if __name__ == '__main__':
    raise SystemExit(main())
