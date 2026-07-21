"""Progress logging to stderr. Silence with FLEETLET_QUIET=1."""

from __future__ import annotations

import os
import sys
import threading

_lock = threading.Lock()


def log(message: str) -> None:
    if os.environ.get("FLEETLET_QUIET"):
        return
    with _lock:
        print(f"[fleetlet] {message}", file=sys.stderr, flush=True)


def log_remote_output(tag: str, stdout: str, stderr: str) -> None:
    """Relay a worker's captured stdout/stderr, prefixed per line."""
    if os.environ.get("FLEETLET_QUIET"):
        return
    with _lock:
        for line in stdout.splitlines():
            print(f"[{tag}] {line}", file=sys.stderr, flush=True)
        for line in stderr.splitlines():
            print(f"[{tag}] {line}", file=sys.stderr, flush=True)
