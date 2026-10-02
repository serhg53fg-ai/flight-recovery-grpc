"""Create clients inside Gunicorn workers; never preload active gRPC channels."""
import os
import re

from flask import jsonify, request
from apps.flight.app import create_app

FILE_ENDPOINTS = frozenset('legacy.' + name for name in (
    'save_scenario', 'get_scenarios', 'delete_scenario', 'upload_optimization_file',
    'run_optimization', 'get_optimization_result', 'download_optimization_result'))
BATCH_ENDPOINTS = frozenset('predictions.' + name for name in ('predict_batch', 'predict_stream', 'upload_predict'))


def create_production_app():
    required = {'FLIGHT_STORAGE_BACKEND': 'mysql', 'FLIGHT_DURABLE_ENABLED': '1',
                'FLIGHT_RECOVERY_ENABLED': '1'}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError('生产HTTP必须显式启用mysql、durable与recovery')
    instance = os.environ.get('FLIGHT_WEB_INSTANCE', 'web')
    if not re.fullmatch('[A-Za-z0-9_-]{1,32}', instance):
        raise ValueError('无效FLIGHT_WEB_INSTANCE')
    app = create_app()
    app.config['PRODUCTION_HTTP'] = True

    @app.before_request
    def disable_local_files():
        if request.endpoint in BATCH_ENDPOINTS:
            return jsonify(success=False, error_code='SYNC_BATCH_DISABLED',
                           error='批量预测请使用后台任务入口'), 410
        if request.endpoint in FILE_ENDPOINTS:
            return jsonify(success=False, error_code='LEGACY_FILES_DISABLED',
                           error='此部署请使用选定预测任务的版本化恢复方案'), 410

    @app.after_request
    def identify_instance(response):
        response.headers['X-Flight-Instance'] = instance
        return response

    return app
