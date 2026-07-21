#!/usr/bin/env python3
"""fleetlet guest runner — the worker server that lives inside each microVM.

This file is copied into the VM (`machine cp`) and launched detached
(`machine exec -d -- python3 /opt/fleetlet/runner.py`). It must stay a
SINGLE SELF-CONTAINED FILE using only the standard library. cloudpickle is
used implicitly when unpickling by-value function blobs (the pickle stream
references it), so it only needs to be installed, never imported here.

Protocol (mirror of _proto.py, PROTO_VERSION 1):
    4-byte big-endian length + pickle(dict) frames over TCP.

Fork model: the host warms this server up (setup done), disconnects, and
freezes the VM as a fork golden. Clones resume inside accept() with the
blob cache, loaded modules, and actor instance already in memory.
"""

import contextlib
import importlib
import importlib.util
import io
import os
import pickle
import socket
import struct
import sys
import time
import traceback

PROTO_VERSION = 1
PICKLE_PROTOCOL = 5
# Must match _proto.GUEST_LOG — on the shared overlay so a separate
# `machine exec ... cat` can read it (per-exec /tmp views differ).
LOG_PATH = "/opt/fleetlet/runner.log"


# ---------------------------------------------------------------- framing

def send_frame(sock, obj):
    data = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_frame(sock):
    header = recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    body = recv_exact(sock, length)
    if body is None:
        raise ConnectionError("EOF mid-frame")
    return pickle.loads(body)


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ---------------------------------------------------------------- state

class Runner:
    def __init__(self, project_root):
        self.project_root = project_root
        self.setup_id = None
        self.blob_cache = {}      # blob_id -> callable
        self.main_modules = {}    # relpath -> module
        self.instance = None      # actor instance for cls pools
        self.log = sys.stderr

    # ------------------------------------------------------------ setup

    def op_hello(self, req):
        return {
            "ok": True,
            "proto": PROTO_VERSION,
            "py": list(sys.version_info[:3]),
            "pid": os.getpid(),
            "cloudpickle": importlib.util.find_spec("cloudpickle") is not None,
            "setup_id": self.setup_id,
        }

    def op_setup(self, req):
        if self.setup_id is not None and req["setup_id"] == self.setup_id:
            return {"ok": True, "cached": True}

        if self.project_root and self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
        for extra in req.get("sys_path", []):
            if extra not in sys.path:
                sys.path.insert(0, extra)

        cls_spec = req.get("cls_spec")
        if cls_spec is not None:
            cls = self.resolve(cls_spec, req.get("cls_blob"))
            args, kwargs = pickle.loads(req["cls_args"])
            self.instance = cls(*args, **kwargs)
            self.run_enter_hooks()

        self.setup_id = req["setup_id"]
        return {"ok": True, "cached": False}

    def run_enter_hooks(self):
        seen = set()
        for klass in reversed(type(self.instance).__mro__):
            for name, attr in vars(klass).items():
                if name in seen or not getattr(attr, "__fleetlet_enter__", False):
                    continue
                seen.add(name)
                getattr(self.instance, name)()

    # ------------------------------------------------------------ resolve

    def resolve(self, spec, blob=None):
        kind = spec["kind"]
        if kind == "blob":
            bid = spec["blob_id"]
            if bid not in self.blob_cache:
                if blob is None:
                    return None  # host must resend with blob attached
                self.blob_cache[bid] = pickle.loads(blob)
            return self.blob_cache[bid]

        if kind == "import":
            obj = importlib.import_module(spec["module"])
        elif kind == "mainfile":
            obj = self.load_main_file(spec["file"])
        elif kind == "self":
            if self.instance is None:
                raise RuntimeError("no actor instance on this worker (cls pool not set up)")
            obj = self.instance
        else:
            raise RuntimeError("unknown fn spec kind: %r" % (kind,))

        for part in spec.get("qualname", "").split("."):
            if part:
                obj = getattr(obj, part)
        return getattr(obj, "_fleetlet_raw", obj)

    def load_main_file(self, relpath):
        """Load the user's entrypoint file and alias it as __main__ so that
        pickled references to __main__.X resolve to the user's classes."""
        if relpath in self.main_modules:
            return self.main_modules[relpath]
        path = os.path.join(self.project_root, relpath)
        spec = importlib.util.spec_from_file_location("__fleetlet_main__", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["__fleetlet_main__"] = module
        spec.loader.exec_module(module)
        sys.modules["__main__"] = module
        self.main_modules[relpath] = module
        return module

    # ------------------------------------------------------------ call

    def op_call(self, req):
        fn = self.resolve(req["spec"], req.get("blob"))
        if fn is None:
            return {"ok": False, "need_blob": True}

        out_buf, err_buf = io.StringIO(), io.StringIO()
        started = time.monotonic()
        try:
            args, kwargs = pickle.loads(req["args"])
            with contextlib.redirect_stdout(Tee(out_buf, self.log)), \
                 contextlib.redirect_stderr(Tee(err_buf, self.log)):
                value = fn(*args, **kwargs)
            value_blob = self.dump_result(value)
            return {
                "ok": True,
                "value": value_blob,
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "duration": time.monotonic() - started,
            }
        except BaseException as exc:  # noqa: BLE001 — everything must cross the wire
            return {
                "ok": False,
                "error": {
                    "etype": type(exc).__name__,
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                    "pickled": self.try_pickle_exc(exc),
                },
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "duration": time.monotonic() - started,
            }

    def dump_result(self, value):
        try:
            return pickle.dumps(value, protocol=PICKLE_PROTOCOL)
        except Exception:
            import cloudpickle  # only for exotic return values
            return cloudpickle.dumps(value, protocol=PICKLE_PROTOCOL)

    def try_pickle_exc(self, exc):
        try:
            return pickle.dumps(exc, protocol=PICKLE_PROTOCOL)
        except Exception:
            return None

    # ------------------------------------------------------------ serve

    def serve(self, host, port):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(8)
        print("fleetlet runner listening on %s:%d (pid %d)" % (host, port, os.getpid()),
              file=self.log, flush=True)
        while True:
            conn, addr = server.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                if self.serve_conn(conn):
                    break
            except (ConnectionError, OSError) as exc:
                print("connection dropped: %r" % (exc,), file=self.log, flush=True)
            finally:
                conn.close()
        server.close()

    def serve_conn(self, conn):
        """Serve one host connection until EOF. Returns True on shutdown op."""
        while True:
            req = recv_frame(conn)
            if req is None:
                return False
            if req.get("op") == "shutdown":
                send_frame(conn, {"ok": True})
                return True
            send_frame(conn, self.dispatch(req))

    def dispatch(self, req):
        """Route one op dict to its handler; never raises. Shared by the
        socket loop and the HTTP frontend."""
        handler = getattr(self, "op_" + str(req.get("op")), None)
        if handler is None:
            return {"ok": False, "error": {"etype": "ProtocolError",
                                           "repr": "unknown op %r" % (req.get("op"),),
                                           "traceback": "", "pickled": None}}
        try:
            return handler(req)
        except BaseException as exc:  # setup errors etc.
            return {
                "ok": False,
                "error": {
                    "etype": type(exc).__name__,
                    "repr": repr(exc),
                    "traceback": traceback.format_exc(),
                    "pickled": self.try_pickle_exc(exc),
                },
            }


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ---------------------------------------------------------------- http frontend

def serve_http(runner, host, port, token):
    """Same ops over HTTP, for workers reached through an HTTPS ingress
    instead of a raw TCP forward (the cloud target).

    One request/response per op won't survive proxy idle timeouts on long
    calls, so ops run async behind a submit/poll pair:

        POST /submit           pickled op dict -> {"id": N}   (409 if busy)
        GET  /poll?id=N&wait=S -> 200 pickled resp | 204 still running
        GET  /healthz          -> 200 "ok"  (unauthenticated; liveness)

    The finished result is RETAINED until the next submit, so a dropped poll
    response can be re-polled. One job at a time — the pool serializes calls
    per worker anyway. Every op request must carry X-Fleetlet-Token.
    """
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = {"seq": 0, "thread": None, "result": None, "lock": threading.Lock()}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(self, status, body, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            if token and self.headers.get("X-Fleetlet-Token") != token:
                self._reply(403, b'{"error": "bad token"}')
                return False
            return True

        def do_GET(self):
            path, _, query = self.path.partition("?")
            if path == "/healthz":
                self._reply(200, b"ok", "text/plain")
                return
            if path != "/poll":
                self._reply(404, b'{"error": "not found"}')
                return
            if not self._authed():
                return
            params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
            job_id = int(params.get("id", -1))
            wait = min(float(params.get("wait", 0)), 25.0)
            with state["lock"]:
                current, thread = state["seq"], state["thread"]
            if job_id != current:
                self._reply(409, json.dumps({"error": "no such job", "id": current}).encode())
                return
            if thread is not None:
                thread.join(wait)
            with state["lock"]:
                if state["thread"] is not None and state["thread"].is_alive():
                    self._reply(204, b"")
                    return
                state["thread"] = None
                result = state["result"]
            self._reply(200, result, "application/octet-stream")

        def do_POST(self):
            if self.path != "/submit":
                self._reply(404, b'{"error": "not found"}')
                return
            if not self._authed():
                return
            length = int(self.headers.get("Content-Length") or 0)
            req = pickle.loads(self.rfile.read(length))
            with state["lock"]:
                if state["thread"] is not None and state["thread"].is_alive():
                    self._reply(409, b'{"error": "busy"}')
                    return
                state["seq"] += 1
                job_id = state["seq"]

                def run():
                    resp = runner.dispatch(req)
                    with state["lock"]:
                        state["result"] = pickle.dumps(resp, protocol=PICKLE_PROTOCOL)

                state["result"] = None
                state["thread"] = threading.Thread(target=run, daemon=True)
                state["thread"].start()
            self._reply(202, json.dumps({"id": job_id}).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    print("fleetlet runner (http) listening on %s:%d (pid %d)" % (host, port, os.getpid()),
          file=runner.log, flush=True)
    server.serve_forever()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--project-root", default="/opt/fleetlet/project")
    parser.add_argument("--http", action="store_true",
                        help="serve ops over HTTP (cloud ingress) instead of raw TCP")
    args = parser.parse_args()

    # Detached exec has no useful console; keep a log inside the guest.
    log = open(LOG_PATH, "a", buffering=1)
    sys.stdout = log
    sys.stderr = log

    runner = Runner(args.project_root)
    runner.log = log
    if args.http:
        # Second auth layer on top of the tenant-authed ingress. Env, not
        # argv: argv is visible guest-wide via /proc.
        serve_http(runner, args.host, args.port,
                   os.environ.get("FLEETLET_RUNNER_TOKEN", ""))
    else:
        runner.serve(args.host, args.port)


if __name__ == "__main__":
    main()
