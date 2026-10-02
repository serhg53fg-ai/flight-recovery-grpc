#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compatibility launcher.

This file keeps backward-compatible startup command:
  python3 llm_server_minimal.py

Internally it delegates to llm_server.py in stdlib mode.
"""

import os

from llm_server import run_stdlib_server


def main() -> None:
    host = os.environ.get("LLM_GATEWAY_HOST", "0.0.0.0")
    port = int(os.environ.get("LLM_GATEWAY_PORT", "8000"))
    run_stdlib_server(host, port)


if __name__ == "__main__":
    main()
