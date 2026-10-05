"""Flask app factory. Inference runs exclusively through the configured gRPC gateway."""
from __future__ import annotations
import os
from pathlib import Path
from flask import Flask, jsonify, request
from apps.flight.clients.inference import InferenceClient
from apps.flight.services.prediction import PredictionService
from apps.flight.storage.factory import create_job_store, storage_environment
from apps.flight.storage.repository import StorageError
from apps.flight.domain.prediction import timezone_for
from apps.flight.tasks.config import durable_environment, validate_durable_config, configured_repository

ROOT = Path(__file__).resolve().parents[2]

def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        RUNTIME_ROOT=os.environ.get('FLIGHT_RUNTIME_ROOT', str(ROOT/'runtime')),
        GATEWAY_ADDRESS=os.environ.get('FLIGHT_GATEWAY_ADDRESS','127.0.0.1:50051'),
        RPC_TIMEOUT_SECONDS=float(os.environ.get('FLIGHT_RPC_TIMEOUT_SECONDS','20')),
        INPUT_TIMEZONE=os.environ.get('FLIGHT_INPUT_TIMEZONE','Asia/Shanghai'),
        BATCH_WORKERS=int(os.environ.get('FLIGHT_BATCH_WORKERS','1')),
        MAX_BATCH_SIZE=1000, MAX_CONTENT_LENGTH=16*1024*1024,
        REPLAY_ROOT=os.environ.get('FLIGHT_REPLAY_ROOT'),
        REPLAY_DATASET_ID=os.environ.get('FLIGHT_REPLAY_DATASET_ID'),
        RELEASE_MANIFEST_PATH=os.environ.get('FLIGHT_RELEASE_MANIFEST_PATH'),
        EXPERIMENTAL_MODEL=os.environ.get('FLIGHT_EXPERIMENTAL_MODEL', '0') == '1',
        REDIS_UNIX_SOCKET=os.environ.get('FLIGHT_REDIS_UNIX_SOCKET'),
    )
    app.config.update(storage_environment())
    app.config.update(durable_environment())
    recovery_enabled = os.environ.get('FLIGHT_RECOVERY_ENABLED', '0')
    if recovery_enabled not in ('0', '1'):
        raise ValueError('FLIGHT_RECOVERY_ENABLED必须为0或1')
    app.config['RECOVERY_ENABLED'] = recovery_enabled == '1'
    if os.environ.get('FLIGHT_EXPERIMENTAL_MODEL', '0') not in ('0', '1'):
        raise ValueError('FLIGHT_EXPERIMENTAL_MODEL必须为0或1')
    if config: app.config.update(config)
    validate_durable_config(app.config)
    if app.config['REPLAY_DATASET_ID'] is not None:
        if not app.config['DURABLE_ENABLED'] or not app.config['REPLAY_ROOT'] or not app.config['RELEASE_MANIFEST_PATH']:
            raise ValueError('历史回放需要持久执行、数据根目录和发布清单')
        from deploy.distributed.release import load_release, release_identity
        identity = release_identity(load_release(Path(app.config['RELEASE_MANIFEST_PATH'])))
        if any(app.config[key] != identity[field] for key, field in (
                ('DURABLE_MODEL_VERSION', 'model_version'),
                ('DURABLE_SOURCE', 'source'),
                ('DURABLE_PROMPT_VERSION', 'prompt_version'))):
            raise ValueError('历史回放发布身份与异步配置不匹配')
        app.config['RELEASE_IDENTITY'] = identity
    if type(app.config['RECOVERY_ENABLED']) is not bool:
        raise ValueError('RECOVERY_ENABLED必须为布尔值')
    if app.config['RECOVERY_ENABLED'] and app.config['STORAGE_BACKEND'] != 'mysql':
        raise ValueError('版本化恢复调度必须使用mysql')
    timezone_for(app.config['INPUT_TIMEZONE'])
    runtime = Path(app.config['RUNTIME_ROOT']).resolve()
    for original in (Path('/srv/flight-example/user/tzb/tiaozhan4'),Path('/srv/flight-example/user/grpc'),Path('/srv/flight-example/user/rpc')):
        if runtime == original or original in runtime.parents:
            raise ValueError('运行目录不能位于原项目目录中')
    for key, name in [('UPLOAD_FOLDER','uploads'),('RESULTS_FOLDER','scenarios'),('OUTPUT_OPT_FOLDER','optimization')]:
        directory=runtime/name
        directory.mkdir(parents=True,exist_ok=True)
        app.config[key]=str(directory)
    store=create_job_store(app.config,runtime)
    durable=configured_repository(app.config)
    if durable is not None:
        from apps.flight.storage.combined import CombinedJobStore
        store=CombinedJobStore(store,durable)
    app.extensions['durable_repository']=durable
    if app.config['RECOVERY_ENABLED']:
        from apps.flight.recovery.repository import RecoveryRepository
        from apps.flight.storage.factory import mysql_config
        app.extensions['recovery_repository'] = RecoveryRepository(mysql_config(app.config))
    client=InferenceClient(app.config['GATEWAY_ADDRESS'],app.config['RPC_TIMEOUT_SECONDS'])
    app.extensions['prediction_client']=client
    app.extensions['job_store']=store
    app.extensions['prediction_service']=PredictionService(client,store,app.config['BATCH_WORKERS'],app.config['MAX_BATCH_SIZE'])
    from apps.flight.api.predictions import predictions
    from apps.flight.api.async_predictions import async_predictions
    from apps.flight.api.replay import replay
    from apps.flight.api.recovery import recovery
    from apps.flight.legacy_routes import legacy
    app.register_blueprint(predictions)
    app.register_blueprint(async_predictions)
    app.register_blueprint(replay)
    app.register_blueprint(recovery)
    app.register_blueprint(legacy)

    @app.get('/health')
    def health():
        return jsonify(status='ok',inference_transport='grpc',fallback_enabled=False,
                       gateway_address=app.config['GATEWAY_ADDRESS'])

    @app.get('/ready')
    def ready():
        from apps.flight.observability import readiness_checks
        checks = readiness_checks(app)
        available = bool(checks) and all(checks.values())
        return jsonify(ready=available, checks=checks), 200 if available else 503

    @app.before_request
    def validate_legacy_names():
        data=request.get_json(silent=True)
        if isinstance(data,dict):
            for key in ('scenario_name','filename'):
                value=data.get(key)
                if value is not None and (not isinstance(value,str) or any(c in value for c in ('/','\\','\x00')) or value in ('.','..')):
                    return jsonify(success=False,error=f'无效 {key}'),400

    @app.errorhandler(413)
    def too_large(_): return jsonify(success=False,error='文件或请求超过 16 MiB'),413

    @app.errorhandler(StorageError)
    def unavailable_storage(_):
        return jsonify(success=False,error_code='STORAGE_UNAVAILABLE',error='任务存储暂不可用'),503
    return app
