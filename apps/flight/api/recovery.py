"""Recovery submission references a terminal prediction job, never a local path."""
from flask import Blueprint, current_app, jsonify, request, Response

from apps.flight.recovery.repository import RecoveryConflict
from apps.flight.recovery.service import build_recovery
from apps.flight.storage.documents import encode, validate_job_id

recovery = Blueprint('recovery', __name__, url_prefix='/api/v1')


def repository():
    repo = current_app.extensions.get('recovery_repository')
    if repo is None:
        return None
    return repo


def disabled():
    return jsonify(success=False, error_code='RECOVERY_DISABLED', error='版本化恢复调度尚未启用'), 503


def payload(plan):
    rid = plan['recovery_id']
    return {**plan, 'success': plan['status'] in {'SUCCEEDED', 'PARTIAL'},
            'status_url': '/api/v1/recovery-jobs/' + rid,
            'download_url': '/api/v1/recovery-jobs/' + rid + '/download'}


@recovery.errorhandler(RecoveryConflict)
def conflict(exc):
    return jsonify(success=False, error_code='IDEMPOTENCY_CONFLICT', error=str(exc)), 409


@recovery.errorhandler(ValueError)
def invalid(exc):
    return jsonify(success=False, error_code='INVALID_ARGUMENT', error=str(exc)), 400


@recovery.errorhandler(KeyError)
def missing(_):
    return jsonify(success=False, error_code='NOT_FOUND', error='未找到预测任务或恢复方案'), 404


@recovery.get('/recovery-capabilities')
def capabilities():
    return jsonify(enabled=repository() is not None, algorithm_version='recovery-greedy-v1',
                   legacy_files_enabled=not current_app.config.get('PRODUCTION_HTTP', False))


@recovery.post('/recovery-jobs')
def submit():
    repo = repository()
    if repo is None:
        return disabled()
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) != {'source_job_id', 'scenario', 'scenario_version'}:
        raise ValueError('需要source_job_id、scenario与scenario_version')
    from apps.flight.recovery.domain import identity
    key = identity(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
    validate_job_id(body['source_job_id'])
    job = current_app.extensions['job_store'].get(body['source_job_id'])
    plan = build_recovery(job, body['scenario'], body['scenario_version'])
    document = repo.save(plan, key)
    response = jsonify(payload(document))
    response.status_code = 201
    response.headers['Location'] = payload(document)['status_url']
    response.headers['Cache-Control'] = 'no-store'
    return response


@recovery.get('/recovery-jobs/<recovery_id>')
def get_job(recovery_id):
    repo = repository()
    if repo is None:
        return disabled()
    return jsonify(payload(repo.get(recovery_id)))


@recovery.get('/recovery-jobs/<recovery_id>/download')
def download(recovery_id):
    repo = repository()
    if repo is None:
        return disabled()
    document = repo.get(recovery_id)
    return Response(encode(document), mimetype='application/json',
                    headers={'Content-Disposition': 'attachment; filename="recovery-' + recovery_id + '.json"', 'Cache-Control': 'no-store'})
