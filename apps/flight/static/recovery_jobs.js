(function () {
    'use strict';
    const panel = document.createElement('section');
    panel.id = 'recoveryPanel';
    panel.style.cssText = 'padding:24px;margin:20px 0;background:#eff6ff;border-radius:12px';
    const container = document.querySelector('.container') || document.body;
    container.prepend(panel);
    if (FlightJobs.recoveryCapabilities && !FlightJobs.recoveryCapabilities.legacy_files_enabled) {
        const legacy = container.querySelector('.content');
        if (legacy) { legacy.hidden = true; legacy.style.display = 'none'; }
    }
    function node(tag, text, parent = panel) {
        const element = document.createElement(tag);
        element.textContent = text || '';
        parent.appendChild(element);
        return element;
    }
    node('h2', '航班智能恢复方案');
    node('p', '基于选定预测任务安排离港/到港事件，独立校验机场容量、串班与过站时间。容量须由场景独立填写，不从预测流量推断。');
    const status = node('p', '请配置恢复时窗与容量');
    status.id = 'recoveryStatus';
    status.setAttribute('role', 'status');
    const source = node('p', '预测任务：' + (FlightJobs.selected() || '未选择'));
    source.id = 'recoverySource';
    const form = node('form');
    const fields = {};
    function input(id, title, value, type = 'text') {
        const label = node('label', title + ' ', form);
        label.style.cssText = 'display:block;margin:10px 0';
        const element = node('input', '', label);
        element.id = id;
        element.type = type;
        element.value = value;
        element.required = true;
        element.style.width = '320px';
        fields[id] = element;
        return element;
    }
    input('recoveryAirport', '目标机场', 'ZGGG');
    input('recoveryStart', '开始时间（完整日期与时区）', '');
    input('recoveryEnd', '结束时间（完整日期与时区）', '');
    input('recoverySlot', '时间槽（分钟）', '15', 'number');
    input('recoveryDepartureCapacity', '每槽离港容量', '10', 'number');
    input('recoveryArrivalCapacity', '每槽到港容量', '8', 'number');
    input('recoveryMtt', '最短过站时间（分钟）', '30', 'number');
    input('recoveryVersion', '场景版本', 'scenario-v1');
    const closureLabel = node('label', '关闭时段（JSON数组；kind为departure或arrival）', form);
    closureLabel.style.display = 'block';
    const closures = node('textarea', '[]', closureLabel);
    closures.id = 'recoveryClosures';
    closures.style.cssText = 'display:block;width:100%;min-height:70px';
    const submit = node('button', '生成恢复方案', form);
    submit.id = 'recoverySubmit';
    submit.type = 'button';
    const download = node('a', '下载完整方案');
    download.id = 'recoveryDownload';
    download.hidden = true;
    const summary = node('p');
    summary.id = 'recoverySummary';
    const metrics = node('p');
    metrics.id = 'recoveryMetrics';
    const situationSummary = node('p');
    situationSummary.id = 'situationSummary';
    const situationRisks = node('div');
    situationRisks.id = 'situationRisks';
    const table = node('table');
    table.id = 'recoveryResults';
    table.style.cssText = 'width:100%;margin-top:15px';
    const header = node('tr', '', node('thead', '', table));
    ['航班标识', '预测离港', '预测到港', '调整后离港', '调整后到港', '新增延误分钟'].forEach(text => node('th', text, header));
    const body = node('tbody', '', table);
    const issues = node('div');
    let busy = false, renderVersion = 0;
    function render(plan) {
        renderVersion += 1;
        status.textContent = `${plan.status} · ${plan.recovery_id}`;
        const sources = Array.from(new Set((plan.provenance || []).map(item => item.source)));
        source.textContent = `预测任务：${plan.source_job_id} · ${sources.join('/')} · 场景：${plan.scenario_version || ''}`;
        if (plan.source_context && plan.source_context.snapshot_hash) {
            source.textContent += ` · 数据集：${plan.source_context.replay_dataset_id || '未知'} · 快照：${plan.source_context.snapshot_hash} · 发布：${plan.source_context.release_identity && plan.source_context.release_identity.release_version || '未知'}`;
        }
        const totals = plan.summary || {};
        summary.textContent = `已安排 ${totals.scheduled_count || 0}，未安排 ${totals.unscheduled_count || 0}，预测排除 ${totals.excluded_prediction_count || 0}；新增延误 ${totals.added_delay_minutes || 0} 分钟，约束违例 ${totals.validation_violation_count || 0}。预测原时刻在同场景下${totals.predicted_baseline_feasible ? '可行' : '不可行或未验证'}；不宣称全局改进。`;
        const validationErrors = Array.isArray(plan.validation_errors) ? plan.validation_errors : null;
        summary.textContent += ` 方案校验：${validationErrors === null ? '未提供' : validationErrors.length ? '未通过' : '通过'}。`;
        const completion = Number(totals.completion_rate || 0) * 100;
        metrics.textContent = `恢复指标：总延误 ${Number(totals.total_delay_minutes || 0).toFixed(1)} 分钟，最大延误 ${Number(totals.max_delay_minutes || 0).toFixed(1)} 分钟，平均延误 ${Number(totals.average_delay_minutes || 0).toFixed(1)} 分钟，受影响航班 ${totals.affected_flight_count || 0}，完成率 ${completion.toFixed(1)}%。`;
        const situation = plan.situation_snapshot || {};
        const prediction = situation.prediction || {};
        situationSummary.textContent = `调度前预测态势：${situation.risk_level || 'UNKNOWN'} · 预测成功 ${prediction.successful || 0}/${prediction.requested || 0} · LLM兜底 ${prediction.flight_fallbacks || 0} · 流量兜底 ${prediction.flow_fallbacks || 0} · 恶劣天气 ${prediction.adverse_weather || 0}`;
        situationRisks.replaceChildren();
        const conflicts = situation.conflicts || [];
        if (!conflicts.length) {
            node('p', '调度前未识别高风险冲突', situationRisks);
        } else {
            const names = {
                DEPARTURE_CAPACITY: '预测累计离港流量超限', ARRIVAL_CAPACITY: '预测累计到港流量超限',
                DEPARTURE_SLOT_CAPACITY: '推出时槽容量超限', ARRIVAL_SLOT_CAPACITY: '上轮挡时槽容量超限',
                AIRPORT_CLOSURE: '机场关闭时段冲突', ROTATION_CONNECTION: '串飞衔接不足',
                ADVERSE_WEATHER: '恶劣天气影响'
            };
            conflicts.forEach(item => {
                const details = [];
                if (item.flight_id) details.push(`航班 ${item.flight_id}`);
                if (item.slot_start && item.slot_end) details.push(`${item.slot_start} → ${item.slot_end}`);
                if (item.count !== undefined && item.capacity !== undefined) details.push(`航班数 ${item.count} / 容量 ${item.capacity}`);
                if (item.horizon_minutes) details.push(`${item.horizon_minutes}分钟窗口`);
                if (item.excess !== undefined) details.push(`超出 ${item.excess}`);
                if (item.shortfall_minutes !== undefined) details.push(`缺口 ${item.shortfall_minutes}分钟`);
                if (item.affected_flights !== undefined) details.push(`影响 ${item.affected_flights} 架`);
                node('p', `${names[item.type] || item.type}${details.length ? '：' + details.join('，') : ''}`, situationRisks);
            });
        }
        body.replaceChildren();
        const inputs = new Map((plan.inputs || []).map(item => [item.flight_id, item]));
        (plan.assignments || []).forEach(item => {
            const row = node('tr', '', body);
            const predicted = inputs.get(item.flight_id) || {};
            [item.flight_id, predicted.predicted_departure || '', predicted.predicted_arrival || '',
                item.departure, item.arrival, item.delay_minutes].forEach(value => node('td', String(value), row));
        });
        issues.replaceChildren();
        (plan.unassigned || []).forEach(item => node('p', `未安排 ${item.flight_id}：${item.reason}`, issues));
        (plan.excluded || []).forEach(item => node('p', `预测输入排除 ${item.flight_id}：${item.reason}`, issues));
        (plan.validation_errors || []).forEach(error => node('p', `校验失败：${error}`, issues));
        download.hidden = !plan.download_url;
        if (plan.download_url) download.href = plan.download_url;
    }
    async function request(url, options) {
        const response = await fetch(url, options);
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || '恢复服务不可用');
        return data;
    }
    async function create() {
        if (busy) return;
        const jobId = FlightJobs.selected();
        if (!jobId) { status.textContent = '请选择预测任务'; return; }
        busy = true;
        submit.disabled = true;
        try {
            if (!crypto.randomUUID) throw new Error('请通过 localhost 隧道或 HTTPS 访问');
            const scenario = {
                airport: fields.recoveryAirport.value.trim().toUpperCase(),
                horizon_start: fields.recoveryStart.value.trim(), horizon_end: fields.recoveryEnd.value.trim(),
                slot_minutes: Number(fields.recoverySlot.value),
                departure_capacity: Number(fields.recoveryDepartureCapacity.value),
                arrival_capacity: Number(fields.recoveryArrivalCapacity.value),
                mtt_minutes: Number(fields.recoveryMtt.value), closures: JSON.parse(closures.value)
            };
            const body = JSON.stringify({source_job_id: jobId, scenario,
                                         scenario_version: fields.recoveryVersion.value.trim()});
            const stored = sessionStorage.getItem('flight.pendingRecovery');
            const prior = stored ? JSON.parse(stored) : null;
            const key = prior && prior.body === body ? prior.key : crypto.randomUUID();
            sessionStorage.setItem('flight.pendingRecovery', JSON.stringify({body, key}));
            status.textContent = '正在生成并校验恢复方案';
            const plan = await request('/api/v1/recovery-jobs', {method: 'POST', body,
                headers: {'Content-Type': 'application/json', 'Idempotency-Key': key}});
            sessionStorage.setItem('flight.selectedRecoveryId', plan.recovery_id);
            sessionStorage.removeItem('flight.pendingRecovery');
            render(plan);
            return plan;
        } catch (error) {
            status.textContent = '生成失败：' + error.message;
        } finally { busy = false; submit.disabled = false; }
    }
    submit.addEventListener('click', create);
    form.addEventListener('submit', event => { event.preventDefault(); create(); });
    window.FlightRecovery = {create, render};
    const selected = sessionStorage.getItem('flight.selectedRecoveryId');
    if (selected) {
        const version = renderVersion;
        request('/api/v1/recovery-jobs/' + encodeURIComponent(selected)).then(plan => {
            if (renderVersion === version && !busy) render(plan);
        }).catch(error => { if (renderVersion === version && !busy) status.textContent = '恢复方案读取失败：' + error.message; });
    }
}());
