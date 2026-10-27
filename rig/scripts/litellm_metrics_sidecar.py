#!/usr/bin/env python3
"""Prometheus exporter for LiteLLM multiprocess metrics.

Workaround for LiteLLM 1.106 --prometheus_metrics_port sidecar returning an
empty body: reads the worker-written .db files from PROMETHEUS_MULTIPROC_DIR
and serves an aggregated scrape on :9093 (loopback only).
"""
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.multiprocess import MultiProcessCollector

DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "/opt/litellm/multiproc")
os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", DIR)
REG = CollectorRegistry()
MultiProcessCollector(REG)


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] in ("/metrics", "/"):
            body = generate_latest(REG)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 9093), H).serve_forever()
