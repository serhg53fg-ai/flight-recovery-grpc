(function () {
    'use strict';
    const caps = FlightJobs.capabilities;
    const terminal = new Set(['SUCCEEDED', 'PARTIAL', 'FAILED', 'EXPIRED', 'CANCELLED']);
    const panel = document.createElement('section');
    panel.id = 'asyncTask';
    panel.style.cssText = 'padding:20px;margin:20px 0;background:#eff6ff;border-radius:12px';
    function element(tag, id, text, parent = panel) {
        const node = document.createElement(tag);
        if (id) node.id = id;
        if (text) node.textContent = text;
        parent.appendChild(node);
        return node;
    }
    element('h3', '', '后台批量预测任务');
    const replayLabel = element('label', '', '历史回放数据集 ');
    const replaySelect = element('select', 'asyncReplayDataset', '', replayLabel);
    replaySelect.disabled = true;
    element('p', 'asyncReplayNote', '正在读取回放数据范围与来源假设');
    const status = element('p', 'asyncTaskStatus', '准备提交；关闭页面不会取消后台任务');
    status.setAttribute('role', 'status');
    const progress = element('p', 'asyncProgress');
    const cancel = element('button', 'asyncCancel', '取消未完成预测');
    cancel.type = 'button';
    cancel.disabled = true;
    const download = element('a', 'asyncDownload', '下载结果');
    download.hidden = true;
    download.style.marginLeft = '16px';
    const results = element('div', 'asyncResults');
    const anchor = document.getElementById('batchResults');
    if (anchor) anchor.parentNode.insertBefore(panel, anchor);
    else document.body.appendChild(panel);
    let current = null, source = null, timer = null, busy = false, version = -1;
    let refreshing = false;
    const rows = new Map();
    const cursorKey = id => 'flight.eventCursor.' + id;
    function stop() {
        if (source) source.close();
        source = null;
        clearInterval(timer);
        timer = null;
    }
    function row(result) {
        const key = result.flight_id;
        if (!key) return;
        let node = rows.get(key);
        if (!node) {
            node = document.createElement('article');
            node.style.cssText = 'margin:12px 0;padding:12px;background:white;border-radius:8px';
            rows.set(key, node);
            results.appendChild(node);
        }
        node.replaceChildren();
        element('strong', '', String(result.flight_no || result.flight_number || key), node);
        if (!result.success) {
            element('p', '', '预测失败：' + (result.error || result.error_code || '未知原因'), node);
            return;
        }
        element('p', '', `预测来源：${result.source || '未知'}；模型：${result.model_version || '未知'}`, node);
        if (result.deployment_stage === 'experimental') {
            element('p', '', '大模型实验部署：当前航班精度尚未通过对照门槛。', node);
        }
        const reasons = {PRIMARY_TIMEOUT: '大模型推理超时', PRIMARY_INVALID_OUTPUT: '大模型输出未通过校验',
            PRIMARY_UNAVAILABLE: '大模型推理不可用', FLOW_MODEL_UNAVAILABLE: '流量模型不可用'};
        if (result.flight_degraded) element('p', '', `航班已使用历史模型兜底：${reasons[result.flight_fallback_reason] || '未知原因'}`, node);
        if (result.flow_degraded) element('p', '', `流量已使用计划值兜底：${reasons[result.flow_fallback_reason] || '未知原因'}`, node);
        const prediction = result.prediction_data || {};
        Object.entries(prediction).forEach(([name, value]) => element('p', '', `预测${name}：${value}`, node));
        const flow = result.airport_flow;
        if (flow && Array.isArray(flow.windows)) {
            element('p', '', '机场流量预测：从同一预测时刻开始的累计窗口；起飞/到达为航班事件预测', node);
            flow.windows.forEach(window => element('p', '',
                `${window.horizon_minutes} 分钟累计（${window.window_start || '预测时刻'} 至 ${window.window_end || '窗口终点'}）：预测起飞 ${window.takeoff}，预测到达 ${window.landing}，预测总计 ${window.total}`, node));
            if (flow.flow_consistency_warning) element('p', '', '提示：累计预测随窗口增长未保持单调，原始预测值已保留。', node);
        }
    }
    function render(job) {
        if (job.job_id !== current) return;
        const nextVersion = Number(job.last_event_id || 0);
        if (nextVersion < version) return;
        version = nextVersion;
        status.textContent = `${job.status} · ${job.job_id}`;
        progress.textContent = `完成 ${job.completed_count || 0}/${job.total_count || 0}；成功 ${job.success_count || 0}，失败 ${job.failed_count || 0}，取消 ${job.cancelled_count || 0}，过期 ${job.expired_count || 0}`;
        const context = job.submission_context;
        if (context && context.replay_dataset_id) {
            document.getElementById('asyncReplayNote').textContent = `任务数据集 ${context.replay_dataset_id}；快照 ${context.snapshot_hash}；发布版本 ${context.release_identity && context.release_identity.release_version || '未知'}`;
        }
        (job.results || []).forEach(row);
        cancel.disabled = terminal.has(job.status);
        download.href = job.download_url || '/api/jobs/' + current + '/download';
        download.hidden = !job.completed_count;
        if (FlightJobs.selected() === current) FlightJobs.record(job);
        if (terminal.has(job.status)) stop();
    }
    async function json(url, options) {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '任务服务不可用');
        return data;
    }
    async function refresh(id) {
        if (refreshing || current !== id) return;
        refreshing = true;
        try { render(await json('/api/v1/prediction-jobs/' + id)); }
        finally { refreshing = false; }
    }
    function connect(id) {
        const after = sessionStorage.getItem(cursorKey(id)) || '0';
        source = new EventSource('/api/v1/prediction-jobs/' + id + '/events?after=' + after);
        const connection = source;
        const handle = async event => {
            if (current !== id || source !== connection) return;
            try {
                const cursor = Number(event.lastEventId || 0);
                if (cursor && cursor <= Number(sessionStorage.getItem(cursorKey(id)) || 0)) return;
                const data = JSON.parse(event.data);
                if (data.result) row(data.result);
                if (cursor) sessionStorage.setItem(cursorKey(id), String(cursor));
                await refresh(id);
            } catch (_error) {
                status.textContent = '暂时无法更新，正在重连；后台任务继续执行';
            }
        };
        ['submitted', 'claimed', 'requeued', 'result', 'cancel_requested', 'complete', 'snapshot'].forEach(kind => connection.addEventListener(kind, handle));
        connection.addEventListener('dependency_error', () => {
            connection.close();
            status.textContent = '存储暂时不可用，等待恢复';
        });
    }
    async function resume(id) {
        stop();
        current = id;
        version = -1;
        rows.clear();
        results.replaceChildren();
        timer = setInterval(async () => {
            try {
                await refresh(id);
                if (timer && (!source || source.readyState === EventSource.CLOSED)) connect(id);
            } catch (_error) { status.textContent = '连接暂时中断，后台任务继续执行'; }
        }, 2000);
        try {
            await refresh(id);
            if (current === id && timer) connect(id);
        } catch (_error) {
            status.textContent = '连接暂时中断，正在恢复后台任务状态';
        }
    }
    function secureCrypto() {
        if (!crypto.subtle || !crypto.randomUUID) throw new Error('请通过 localhost 隧道或 HTTPS 访问预测页面');
    }
    async function submit(url, body, fingerprint, headers = {}) {
        if (busy) throw new Error('正在提交，请稍候');
        busy = true;
        try {
            secureCrypto();
            const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify([caps, url, fingerprint])));
            const hash = Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
            const stored = sessionStorage.getItem('flight.pendingSubmission');
            const pending = stored ? JSON.parse(stored) : null;
            const key = pending && pending.hash === hash ? pending.key : crypto.randomUUID();
            sessionStorage.setItem('flight.pendingSubmission', JSON.stringify({hash, key}));
            const response = await fetch(url, {method: 'POST', body, headers: {...headers, 'Idempotency-Key': key}});
            const job = await response.json();
            if (response.status !== 202 || !job.accepted || !job.job_id) throw new Error(job.error || '任务未被接受');
            sessionStorage.removeItem('flight.pendingSubmission');
            FlightJobs.record(job);
            // Acceptance is authoritative even if the subsequent status request fails.
            try { await resume(job.job_id); }
            catch (_error) { render(job); }
            return job;
        } catch (error) {
            status.textContent = '提交失败：' + error.message + '；重试相同输入将复用提交键';
            throw error;
        } finally { busy = false; }
    }
    function submitFlights(flights) {
        const body = JSON.stringify({flights, input_timezone: caps.input_timezone,
            ...(replaySelect.value ? {replay_dataset_id: replaySelect.value} : {})});
        return submit('/api/v1/prediction-jobs', body, body, {'Content-Type': 'application/json'});
    }
    async function submitFile(file) {
        if (!file) throw new Error('请先选择 Excel 文件');
        secureCrypto();
        const bytes = await file.arrayBuffer();
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        const body = new FormData();
        body.append('file', file);
        body.append('input_timezone', caps.input_timezone);
        if (replaySelect.value) body.append('replay_dataset_id', replaySelect.value);
        return submit('/api/v1/prediction-jobs/upload', body, Array.from(new Uint8Array(digest)));
    }
    cancel.addEventListener('click', async () => {
        const id = current;
        cancel.disabled = true;
        try { render(await json('/api/v1/prediction-jobs/' + id + '/cancel', {method: 'POST'})); }
        catch (error) { if (current === id) { cancel.disabled = false; status.textContent = error.message; } }
    });
    window.processJsonData = function () {
        return submitFlights(parseJsonInput(document.getElementById('jsonInput').value));
    };
    window.uploadAndPredict = function () {
        const file = typeof selectedFile !== 'undefined' ? selectedFile : document.querySelector('input[type=file]').files[0];
        return submitFile(file);
    };
    window.FlightAsync = {submitFlights, submitFile, resume};
    fetch('/api/v1/replay-datasets').then(response => response.json()).then(data => {
        const datasets = data.datasets || [];
        replaySelect.replaceChildren();
        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset.dataset_id;
            option.textContent = `${dataset.dataset_id} · ${dataset.airport} · ${dataset.coverage_start} 至 ${dataset.coverage_end}`;
            replaySelect.appendChild(option);
        });
        replaySelect.disabled = datasets.length === 0;
        replaySelect.value = datasets.some(dataset => dataset.dataset_id === caps.replay_dataset_id) ? caps.replay_dataset_id : (datasets[0] && datasets[0].dataset_id || '');
        document.getElementById('asyncReplayNote').textContent = datasets.length ?
            '历史回放；计划表修订时刻未知，事件接收时间按回放假设。数值模型尚未证明天气增益。' :
            '未配置历史回放数据集；当前任务不会自动补充机场上下文。';
    }).catch(() => { document.getElementById('asyncReplayNote').textContent = '回放数据集列表暂不可用'; });
    const selected = FlightJobs.selected();
    if (selected) resume(selected).catch(() => { status.textContent = '未找到所选异步任务，请提交新任务'; });
}());
