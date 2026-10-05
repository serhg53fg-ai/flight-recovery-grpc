(function () {
    'use strict';

    const STORAGE_KEY = 'flight.selectedJobId';
    const JOB_SCOPED_PATHS = new Set([
        '/api/latest-results',
        '/api/predictions',
        '/api/flight-data',
        '/api/search-flight',
        '/api/timeline-data',
        '/api/timeline-data-with-scenario'
    ]);
    const RESULT_PATHS = new Set([
        '/predict_flight',
        '/predict_flights_batch',
        '/upload',
        '/api/excel-batch-predict'
    ]);
    const nativeFetch = window.fetch.bind(window);

    function validJobId(value) {
        return typeof value === 'string' &&
            /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
    }

    function selected() {
        const queryValue = new URL(window.location.href).searchParams.get('job_id');
        if (validJobId(queryValue)) {
            sessionStorage.setItem(STORAGE_KEY, queryValue);
            return queryValue;
        }
        const stored = sessionStorage.getItem(STORAGE_KEY);
        return validJobId(stored) ? stored : null;
    }

    function sourceValues(data) {
        const values = [];
        if (typeof data.source === 'string') values.push(data.source);
        if (Array.isArray(data.sources)) values.push(...data.sources);
        if (Array.isArray(data.results)) {
            data.results.forEach(item => {
                if (item && typeof item.source === 'string') values.push(item.source);
            });
        }
        return Array.from(new Set(values));
    }

    function sourceBanner() {
        let banner = document.getElementById('prediction-source');
        if (banner) return banner;
        banner = document.createElement('div');
        banner.id = 'prediction-source';
        banner.setAttribute('role', 'status');
        Object.assign(banner.style, {
            position: 'sticky',
            top: '0',
            zIndex: '10000',
            padding: '10px 16px',
            borderBottom: '1px solid #bfdbfe',
            background: '#eff6ff',
            color: '#1e3a8a',
            fontWeight: '700',
            textAlign: 'center',
            boxShadow: '0 2px 8px rgba(15, 23, 42, 0.08)'
        });
        document.body.insertBefore(banner, document.body.firstChild);
        return banner;
    }

    function render(data) {
        const banner = sourceBanner();
        const jobId = data.job_id || selected();
        const sources = sourceValues(data);
        const notice = document.getElementById('model-notice');
        if (notice && sources.length) {
            const versions = Array.from(new Set([data.model_version, ...(Array.isArray(data.results) ? data.results : []).map(row => row && row.model_version)]
                .filter(value => typeof value === 'string' && value.length)));
            const label = sources.map(source => ({TEST: 'TEST 测试后端', LLM: '大模型推理', BASELINE: '历史模型兜底'}[source] || source)).join(' / ');
            notice.textContent = `本次任务实际来源：${label}；版本：${versions.join(' / ') || '未提供'}。` +
                (sources.includes('TEST') ? 'TEST 结果仅验证工程链路，不代表真实模型精度。' : '预测结果需结合场景与模型评估使用。');
        } else if (notice) {
            notice.textContent = data.success === false ? '本次预测失败，暂无可确认的模型来源。' : '等待本次任务返回实际模型来源与版本。';
        }
        const shortId = jobId ? jobId.slice(0, 8) : '未选择';
        if (sources.includes('TEST')) {
            banner.textContent = `当前任务 ${shortId} · TEST 测试后端 · 结果不代表真实 Qwen 推理`;
            banner.style.background = '#fff7ed';
            banner.style.borderBottomColor = '#fdba74';
            banner.style.color = '#9a3412';
        } else if (sources.includes('LLM')) {
            banner.textContent = `当前任务 ${shortId} · Qwen LLM · ${data.status || '推理结果'}`;
            banner.style.background = '#ecfdf5';
            banner.style.borderBottomColor = '#86efac';
            banner.style.color = '#166534';
        } else if (sources.includes('BASELINE')) {
            banner.textContent = `当前任务 ${shortId} · 历史基线模型 · ${data.status || '预测结果'}`;
            banner.style.background = '#eff6ff';
            banner.style.borderBottomColor = '#93c5fd';
            banner.style.color = '#1e40af';
        } else if (jobId && data.success === false) {
            banner.textContent = `当前任务 ${shortId} · 预测失败`;
        } else {
            banner.textContent = jobId ? `当前任务 ${shortId} · 等待预测来源` : '尚未选择预测任务';
        }
    }

    function updateLinks() {
        const jobId = selected();
        if (!jobId) return;
        document.querySelectorAll('a[href]').forEach(link => {
            const url = new URL(link.href, window.location.href);
            if (url.origin !== window.location.origin) return;
            if (!['/timeline', '/overview', '/scenario'].includes(url.pathname)) return;
            url.searchParams.set('job_id', jobId);
            link.href = url.pathname + url.search + url.hash;
        });
    }

    function jobUrl(path) {
        const url = new URL(path, window.location.href);
        const jobId = selected();
        if (url.origin === window.location.origin && jobId) url.searchParams.set('job_id', jobId);
        return url.pathname + url.search + url.hash;
    }

    function record(data) {
        if (!data || typeof data !== 'object') return;
        if (validJobId(data.job_id)) {
            sessionStorage.setItem(STORAGE_KEY, data.job_id);
            const url = new URL(window.location.href);
            if (url.searchParams.has('job_id')) {
                url.searchParams.set('job_id', data.job_id);
                history.replaceState(null, '', url.pathname + url.search + url.hash);
            }
        }
        updateLinks();
        render(data);
    }

    function withSelectedJob(input) {
        const urlText = typeof input === 'string' || input instanceof URL ? input.toString() : input.url;
        const url = new URL(urlText, window.location.href);
        const jobId = selected();
        if (url.origin === window.location.origin && JOB_SCOPED_PATHS.has(url.pathname) && jobId) {
            url.searchParams.set('job_id', jobId);
            if (input instanceof Request) return new Request(url.toString(), input);
            return url.toString();
        }
        return input;
    }

    window.fetch = async function (input, init) {
        const scopedInput = withSelectedJob(input);
        const response = await nativeFetch(scopedInput, init);
        const urlText = typeof scopedInput === 'string' || scopedInput instanceof URL ?
            scopedInput.toString() : scopedInput.url;
        const url = new URL(urlText, window.location.href);
        const contentType = response.headers.get('content-type') || '';
        if (url.origin === window.location.origin && contentType.includes('application/json') &&
            (RESULT_PATHS.has(url.pathname) || url.pathname.startsWith('/api/jobs/'))) {
            try {
                record(await response.clone().json());
            } catch (_error) {
                // The caller remains responsible for reporting malformed responses.
            }
        }
        return response;
    };

    async function* events(response) {
        if (!response.body) throw new Error('当前浏览器不支持流式响应');
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        try {
            while (true) {
                const result = await reader.read();
                buffer += decoder.decode(result.value || new Uint8Array(), {stream: !result.done});
                const blocks = buffer.split(/\r?\n\r?\n/);
                buffer = blocks.pop() || '';
                for (const block of blocks) {
                    const payload = block.split(/\r?\n/)
                        .filter(line => line.startsWith('data:'))
                        .map(line => line.slice(5).trimStart())
                        .join('\n');
                    if (payload) yield JSON.parse(payload);
                }
                if (result.done) break;
            }
            if (buffer.trim()) {
                const payload = buffer.split(/\r?\n/)
                    .filter(line => line.startsWith('data:'))
                    .map(line => line.slice(5).trimStart())
                    .join('\n');
                if (payload) yield JSON.parse(payload);
            }
        } finally {
            reader.releaseLock();
        }
    }

    async function initialize() {
        if (window.location.pathname === '/scenario') {
            try {
                const response = await nativeFetch('/api/v1/recovery-capabilities');
                const capabilities = await response.json();
                if (response.ok && capabilities.enabled) {
                    window.FlightJobs.recoveryCapabilities = capabilities;
                    const script = document.createElement('script');
                    script.src = '/static/recovery_jobs.js';
                    document.head.appendChild(script);
                }
            } catch (_error) { /* Explicit recovery capability remains unavailable. */ }
        }
        if (window.location.pathname === '/flight_input') {
            try {
                const response = await nativeFetch('/api/v1/prediction-capabilities');
                const capabilities = await response.json();
                if (response.ok && capabilities.enabled) {
                    window.FlightJobs.capabilities = capabilities;
                    const script = document.createElement('script');
                    script.src = '/static/async_predictions.js';
                    document.head.appendChild(script);
                }
            } catch (_error) { /* Existing synchronous controls remain available. */ }
        }
        const jobId = selected();
        updateLinks();
        render({job_id: jobId});
        if (!jobId) return;
        try {
            const response = await nativeFetch(`/api/jobs/${encodeURIComponent(jobId)}`);
            if (response.ok) {
                const job = await response.json();
                if (selected() === jobId) record(job);
            }
        } catch (_error) {
            if (selected() === jobId) render({job_id: jobId});
        }
    }

    window.FlightJobs = {selected, record, updateLinks, events, jobUrl};
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize, {once: true});
    } else {
        initialize();
    }
}());
