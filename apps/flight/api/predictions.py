"""Prediction endpoints and explicit job-scoped result views."""
from __future__ import annotations
import io
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from flask import Blueprint, current_app, jsonify, request, Response, send_file, stream_with_context
from apps.flight.services.prediction import serializable
from apps.flight.storage.repository import StorageError

predictions=Blueprint('predictions',__name__)
HTTP_ERRORS={'INVALID_ARGUMENT':400,'UNAVAILABLE':503,'RESOURCE_EXHAUSTED':429,
             'DEADLINE_EXCEEDED':504,'CANCELLED':409,'DATA_LOSS':502,'INTERNAL':500}
COLUMNS=['航班号','机尾号','机型','性质','起飞站四字码','到达站四字码','计划离港时间','计划到港时间',
         '预测实际离港时间','预测实际起飞时间','预测实际落地时间','预测实际到港时间',
         '预测离港延误(分钟)','预测到港延误(分钟)','机场流量预测','天气过期','航班降级','流量降级',
         '数据版本','Prompt版本','预测来源','trace_id','flight_id','模型版本',
         '数据集ID','快照哈希','发布版本','航班兜底原因','流量兜底原因','部署阶段','航班精度门槛通过']

def service(): return current_app.extensions['prediction_service']
def store(): return current_app.extensions['job_store']
def input_timezone():
    body=request.get_json(silent=True)
    return (body.get('input_timezone') if isinstance(body,dict) else None) or request.form.get('input_timezone') or current_app.config['INPUT_TIMEZONE']

def selected_job_id():
    body=request.get_json(silent=True)
    value=request.args.get('job_id') or (body.get('job_id') if isinstance(body,dict) else None)
    if not value: raise ValueError('请先选择预测任务，缺少 job_id')
    return value

def rows_for_job(job):
    rows=[]
    submission = job.get('submission_context') or {}
    release = submission.get('release_identity') or {}
    for row in job['results']:
        if not row['success']: continue
        info=row['flight_info']; p=row['prediction_data']
        rows.append({'航班号':info['航班号'],'机尾号':info['机尾号'],'机型':info['机型'],
            '性质':info.get('性质', ''),
            '起飞站四字码':info['计划起飞站四字码'],'到达站四字码':info['计划到达站四字码'],
            '计划离港时间':info['计划离港时间'],'计划到港时间':info['计划到港时间'],
            **{'预测'+k:v for k,v in p.items()},
            '预测离港延误(分钟)':(datetime.fromisoformat(p['实际离港时间'])-datetime.fromisoformat(info['计划离港时间'])).total_seconds()/60,
            '预测到港延误(分钟)':(datetime.fromisoformat(p['实际到港时间'])-datetime.fromisoformat(info['计划到港时间'])).total_seconds()/60,
            '机场流量预测':json.dumps(row.get('airport_flow'), ensure_ascii=False) if row.get('airport_flow') else '',
            '天气过期':row.get('weather_stale', False), '航班降级':row.get('flight_degraded', False),
            '流量降级':row.get('flow_degraded', False),
            '数据版本':row.get('data_version', ''), 'Prompt版本':row.get('prompt_version', ''),
            '预测来源':row['source'],'trace_id':row['trace_id'],'flight_id':row['flight_id'],'模型版本':row['model_version'],
            '数据集ID':submission.get('replay_dataset_id', ''),
            '快照哈希':submission.get('snapshot_hash', ''),
            '发布版本':release.get('release_version', ''),
            '航班兜底原因':row.get('flight_fallback_reason', ''),
            '流量兜底原因':row.get('flow_fallback_reason', ''),
            '部署阶段':row.get('deployment_stage', ''),
            '航班精度门槛通过':row.get('flight_gate_passed')})
    return rows

def frame_for_job(job): return pd.DataFrame(rows_for_job(job),columns=COLUMNS)

def job_payload(job):
    return {**job,'success':job['success_count']>0,'total_processed':job['total_count'],
            'predictions':rows_for_job(job),'result_file':job['job_id']+'.xlsx',
            'download_url':'/api/jobs/'+job['job_id']+'/download','file_time':job['created_at']}

@predictions.errorhandler(ValueError)
def invalid(exc): return jsonify(success=False,error_code='INVALID_ARGUMENT',error=str(exc)),400

@predictions.errorhandler(KeyError)
def missing(exc): return jsonify(success=False,error='没有找到该任务'),404

@predictions.post('/predict_flight')
def predict_flight():
    body=request.get_json(silent=True)
    if not isinstance(body,dict) or not body: raise ValueError('请输入航班 JSON 对象')
    data=body.get('flight',body)
    job=service().predict_many([data],input_timezone())
    result=job['results'][0]
    payload={**result,'status':job['status'],'fallback_enabled':False,'fallback_reason':job['fallback_reason']}
    return jsonify(payload),200 if result['success'] else HTTP_ERRORS.get(result['error_code'],502)


def batch_input():
    body=request.get_json(silent=True)
    flights=body.get('flights') if isinstance(body,dict) else body
    if not isinstance(flights,list) or not flights: raise ValueError('请输入非空 flights 数组')
    return flights

@predictions.post('/predict_flights_batch')
def predict_batch():
    return jsonify(job_payload(service().predict_many(batch_input(),input_timezone())))

@predictions.post('/predict_flights_stream')
def predict_stream():
    flights=batch_input()
    svc=service()
    job=svc.start(flights,input_timezone())
    def generate():
        iterator=svc.iter_results(flights,job)
        completed=success=0
        try:
            for row in iterator:
                completed+=1;success+=int(row['success'])
                row={**row,'progress':{'completed':completed,'total':len(flights),'success':success,'failed':completed-success}}
                yield 'data: '+json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n\n'
            final=svc.store.get(job['job_id'])
            yield 'data: '+json.dumps({'type':'complete','job_id':job['job_id'],'status':final['status'],
                'sources':sorted({row['source'] for row in final['results'] if row['success']}),
                'summary':{'total':len(flights),'success':final['success_count'],'failed':final['failed_count']}},ensure_ascii=False)+'\n\n'
        except StorageError:
            yield 'data: '+json.dumps({'type':'error','job_id':job['job_id'],
                'error_code':'STORAGE_UNAVAILABLE','error':'结果尚未持久保存，请稍后查询任务'},ensure_ascii=False)+'\n\n'
        finally:
            iterator.close()
    return Response(stream_with_context(generate()),mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


def read_spreadsheet():
    upload=request.files.get('file')
    if upload is None or not upload.filename: raise ValueError('没有选择文件')
    suffix=Path(upload.filename).suffix.lower()
    if suffix not in ('.xlsx','.xls','.csv'): raise ValueError('支持 .xlsx、.xls、.csv')
    limit=current_app.config['MAX_BATCH_SIZE']
    try:
        if suffix=='.csv': df=pd.read_csv(upload,nrows=limit+1)
        else: df=pd.read_excel(upload,engine='openpyxl' if suffix=='.xlsx' else 'xlrd',nrows=limit+1)
    except Exception as exc: raise ValueError('文件格式或内容无法读取') from exc
    if not 1<=len(df)<=limit: raise ValueError(f'文件行数必须为 1—{limit}')
    records=[]
    for raw in df.to_dict('records'):
        records.append({k:(None if pd.isna(v) else serializable(v)) for k,v in raw.items()})
    return records,list(df.columns)

@predictions.post('/upload')
@predictions.post('/api/excel-batch-predict')
def upload_predict():
    records,_=read_spreadsheet()
    return jsonify(job_payload(service().predict_many(records,input_timezone())))

@predictions.post('/api/preview-excel')
def preview_excel():
    records,columns=read_spreadsheet()
    return jsonify(success=True,preview=records[:5],columns=columns,total_rows=len(records))

@predictions.get('/api/jobs')
def jobs(): return jsonify(success=True,jobs=store().list_jobs())

@predictions.get('/api/jobs/<job_id>')
def get_job(job_id): return jsonify(job_payload(store().get(job_id)))

@predictions.get('/api/jobs/<job_id>/download')
@predictions.get('/download/<job_id>.xlsx')
def download(job_id):
    job=store().get(job_id)
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        frame_for_job(job).to_excel(writer,index=False,sheet_name='预测结果')
        failures=[{'index':r['index'],'trace_id':r['trace_id'],'error_code':r['error_code'],'error':r['error']}
                  for r in job['results'] if not r['success']]
        if failures: pd.DataFrame(failures).to_excel(writer,index=False,sheet_name='失败记录')
    output.seek(0)
    return send_file(output,as_attachment=True,download_name=job_id+'.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@predictions.get('/api/latest-results')
def selected_or_jobs():
    # No server-global last job: this compatibility endpoint only reads an explicit selection.
    if request.args.get('job_id'): return jsonify(job_payload(store().get(selected_job_id())))
    return jsonify(success=True,jobs=store().list_jobs(),message='请选择任务')

@predictions.get('/api/predictions')
def selected_predictions(): return jsonify(rows_for_job(store().get(selected_job_id())))

@predictions.get('/api/flight-data')
def selected_flights():
    job=store().get(selected_job_id())
    rows=rows_for_job(job)
    return jsonify(success=True,job_id=job['job_id'],flights=rows,total_count=len(rows),file_time=job['created_at'])

@predictions.post('/api/search-flight')
def search():
    body=request.get_json(silent=True) or {}
    number=body.get('flight_number','')
    if not isinstance(number,str) or not number.strip(): raise ValueError('请输入航班号')
    rows=[r for r in rows_for_job(store().get(selected_job_id())) if number.upper() in r['航班号'].upper()]
    return jsonify(success=bool(rows),flights=rows,count=len(rows))


def timeline_payload(job,scenario_name=None):
    from apps.flight.legacy_routes import process_timeline_data,apply_stop_periods_with_capacity
    df=frame_for_job(job)
    if not len(df): raise ValueError('该任务没有成功预测，无法生成时间轴')
    if scenario_name:
        path=Path(current_app.config['RESULTS_FOLDER'])/f'scenario_{scenario_name}.json'
        if not path.is_file(): raise KeyError('没有找到情景')
        scenario=json.loads(path.read_text())
        df=apply_stop_periods_with_capacity(df,scenario['stop_periods'],scenario.get('hourly_capacity',60))
    return dict(success=True,job_id=job['job_id'],scenario_applied=bool(scenario_name),
                sources=sorted({r['source'] for r in job['results'] if r['success']}),
                **process_timeline_data(df))

@predictions.get('/api/jobs/<job_id>/timeline')
def job_timeline(job_id): return jsonify(timeline_payload(store().get(job_id)))

@predictions.get('/api/timeline-data')
def selected_timeline(): return jsonify(timeline_payload(store().get(selected_job_id())))

@predictions.post('/api/timeline-data-with-scenario')
def scenario_timeline():
    body=request.get_json(silent=True) or {}
    return jsonify(timeline_payload(store().get(selected_job_id()),body.get('scenario_name')))
