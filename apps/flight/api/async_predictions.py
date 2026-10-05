"""Async acceptance and views; execution never belongs to a Web request."""
from flask import Blueprint, current_app, jsonify, request, Response, stream_with_context

from apps.flight.api.predictions import job_payload, read_spreadsheet
from apps.flight.tasks.documents import AdmissionFull, IdempotencyConflict, label

async_predictions = Blueprint('async_predictions', __name__, url_prefix='/api/v1')


class AsyncDisabled(RuntimeError):
    pass


def repository():
    repo = current_app.extensions.get('durable_repository')
    if repo is None:
        raise AsyncDisabled()
    return repo


def payload(job, accepted=False):
    base = job_payload({k: v for k, v in job.items() if k != 'flights'})
    job_id = job['job_id']
    return {**base, 'accepted': accepted, 'success': True if accepted else base['success'],
            'status_url': '/api/v1/prediction-jobs/' + job_id,
            'events_url': '/api/v1/prediction-jobs/' + job_id + '/events',
            'cancel_url': '/api/v1/prediction-jobs/' + job_id + '/cancel'}


@async_predictions.errorhandler(AsyncDisabled)
def disabled(_):
    return jsonify(success=False, error_code='ASYNC_DISABLED', error='持久异步预测尚未启用'), 503


@async_predictions.errorhandler(IdempotencyConflict)
def conflict(_):
    return jsonify(success=False, error_code='IDEMPOTENCY_CONFLICT', error='此提交键已绑定不同输入或部署配置'), 409


@async_predictions.errorhandler(AdmissionFull)
def backlog(_):
    return jsonify(success=False, error_code='QUEUE_BACKLOG', error='等待任务已达到上限，请稍后提交'), 429


@async_predictions.errorhandler(ValueError)
def invalid(exc):
    return jsonify(success=False, error_code='INVALID_ARGUMENT', error=str(exc)), 400


@async_predictions.errorhandler(KeyError)
def missing(_):
    return jsonify(success=False, error_code='NOT_FOUND', error='没有找到持久任务'), 404


@async_predictions.get('/prediction-capabilities')
def capabilities():
    cfg = current_app.config
    if not cfg['DURABLE_ENABLED']:
        return jsonify(enabled=False)
    return jsonify(enabled=True, model_version=cfg['DURABLE_MODEL_VERSION'], source=cfg['DURABLE_SOURCE'],
                   prompt_version=cfg['DURABLE_PROMPT_VERSION'], input_timezone=cfg['INPUT_TIMEZONE'],
                   max_batch=cfg['MAX_BATCH_SIZE'], replay_dataset_id=cfg.get('REPLAY_DATASET_ID'))


def submit(flights, timezone, replay_dataset_id=None):
    repo = repository()
    cfg = current_app.config
    key = label(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
    from apps.flight.api.replay import prepare_submission
    flights, submission_context = prepare_submission(flights, timezone, replay_dataset_id)
    job = repo.submit(flights, timezone, key, cfg['DURABLE_MODEL_VERSION'], cfg['DURABLE_SOURCE'],
                      cfg['DURABLE_PROMPT_VERSION'], cfg['DURABLE_TTL_SECONDS'], cfg['DURABLE_MAX_ATTEMPTS'],
                      submission_context=submission_context)
    response = jsonify(payload(job, accepted=True))
    response.status_code = 202
    response.headers['Location'] = '/api/v1/prediction-jobs/' + job['job_id']
    response.headers['Cache-Control'] = 'no-store'
    return response


@async_predictions.post('/prediction-jobs')
def submit_json():
    repository()
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or set(body) - {'flights', 'input_timezone', 'replay_dataset_id'}:
        raise ValueError('请输入flights对象，只允许flights、input_timezone和replay_dataset_id')
    return submit(body.get('flights'), body.get('input_timezone', current_app.config['INPUT_TIMEZONE']),
                  body.get('replay_dataset_id'))


@async_predictions.post('/prediction-jobs/upload')
def submit_upload():
    repository()
    label(request.headers.get('Idempotency-Key'), 'Idempotency-Key')
    records, _ = read_spreadsheet()
    return submit(records, request.form.get('input_timezone') or current_app.config['INPUT_TIMEZONE'],
                  request.form.get('replay_dataset_id'))


@async_predictions.get('/prediction-jobs')
def jobs():
    return jsonify(success=True, jobs=repository().list_jobs())


@async_predictions.get('/prediction-jobs/<job_id>')
def get_job(job_id):
    return jsonify(payload(repository().get(job_id)))


@async_predictions.post('/prediction-jobs/<job_id>/cancel')
def cancel(job_id):
    return jsonify(payload(repository().cancel(job_id)))


@async_predictions.get('/prediction-jobs/<job_id>/events')
def events(job_id):
    from .event_stream import parse_cursor, stream_events
    repo = repository()
    cursor = parse_cursor(request.headers, request.args)
    # Fail before streaming response headers have been committed.
    repo.read_event_page(job_id, cursor)
    cfg = current_app.config
    iterator = stream_events(repo, job_id, cursor, cfg['SSE_POLL_SECONDS'],
                             cfg['SSE_HEARTBEAT_SECONDS'], cfg['SSE_MAX_SECONDS'])
    return Response(stream_with_context(iterator), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache, no-store', 'X-Accel-Buffering': 'no'})
