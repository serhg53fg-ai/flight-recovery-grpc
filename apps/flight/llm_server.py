#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified LLM gateway with proper argparse support.
"""

from __future__ import annotations

import json
import os
import time
import uuid
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, List

try:
    from llm_flight_predictor import llm_predictor
except Exception:
    llm_predictor = None

FASTAPI_AVAILABLE = True
try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception:
    FASTAPI_AVAILABLE = False


def _run_single_prediction(flight_data: Dict[str, Any], fallback_to_simulation: bool,) -> Dict[str, Any]:
    request_id = str(uuid.uuid4())
    start = time.time()
    prediction = None
    source = "stdlib_simulation"
    err = None

    if llm_predictor is not None:
        try:
            prediction = llm_predictor.predict(flight_data)
            source = "llm" if prediction is not None else "stdlib_simulation"
            if prediction is None and fallback_to_simulation:
                prediction = llm_predictor.simulate_prediction(flight_data)
                source = "simulation_fallback"
        except Exception as e:
            err = str(e)

    if prediction is None:
        now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
        prediction = {"实际离港时间": now, "实际起飞时间": now, "实际落地时间": now, "实际到港时间": now,}

    latency_ms = int((time.time() - start) * 1000)
    return {"request_id": request_id, "success": True, "source": source, "latency_ms": latency_ms, "prediction": prediction, "error": err,}


def _health_payload() -> Dict[str, Any]:
    model_loaded = bool(getattr(llm_predictor, "model_loaded", False)) if llm_predictor else False
    dep_status = {}
    runtime_status = {}
    if llm_predictor and hasattr(llm_predictor, "dependency_status"):
        dep_status = llm_predictor.dependency_status()
    if llm_predictor and hasattr(llm_predictor, "runtime_status"):
        runtime_status = llm_predictor.runtime_status()
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": getattr(llm_predictor, "model_path", None) if llm_predictor else None,
        "last_error": getattr(llm_predictor, "last_error", None) if llm_predictor else "llm_predictor unavailable",
        "mode": "fastapi" if FASTAPI_AVAILABLE else "stdlib",
        "dependencies": dep_status,
        "runtime": runtime_status,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json_response(self, status_code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _read_json_body(self) -> Dict[str, Any]:
        length_str = self.headers.get("Content-Length", "0")
        try:
            length = int(length_str)
        except ValueError:
            length = 0
        raw = self.rfile.read(length).decode("utf-8", errors="ignore") if length > 0 else "{}"
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        if self.path in ["/", ""]:
            self._json_response(200, {"service": "llm-gateway", "status": "ok", "version": "2.0.0"})
            return
        if self.path == "/health":
            self._json_response(200, _health_payload())
            return
        self._json_response(404, {"error": "not found", "path": self.path})

    def do_POST(self):
        data = self._read_json_body()
        if self.path == "/predict":
            flight_data = data.get("flight_data", {}) if isinstance(data, dict) else {}
            fallback = bool(data.get("fallback_to_simulation", True)) if isinstance(data, dict) else True
            self._json_response(200, _run_single_prediction(flight_data, fallback))
            return
        self._json_response(200, _run_single_prediction(data if isinstance(data, dict) else {}, True))

    def log_message(self, fmt, *args):
        import sys
        print("[llm-gateway] %s - %s" % (self.address_string(), fmt % args), file=sys.stderr)


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def run_stdlib_server(host: str, port: int) -> None:
    with ThreadingHTTPServer((host, port), Handler) as httpd:
        print(f"LLM gateway(stdlib) listening on {host}:{port}")
        httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Flight Gateway")
    parser.add_argument("--host", default=os.environ.get("LLM_GATEWAY_HOST", "0.0.0.0"), help="Server host")
    parser.add_argument("--port", type=int, default=int(os.environ.get("LLM_GATEWAY_PORT", "8000")), help="Server port")
    parser.add_argument("--mode", default=os.environ.get("LLM_SERVER_MODE", "auto").lower(), help="Server mode")
    args = parser.parse_args()

    print(f"Starting LLM gateway on {args.host}:{args.port} (mode={args.mode})")
    run_stdlib_server(args.host, args.port)


if __name__ == "__main__":
    main()
