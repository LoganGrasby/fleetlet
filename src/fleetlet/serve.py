"""HTTP gateway — an App's functions as a JSON/HTTP service.

FastMCP's bargain, applied to VMs: declaring the function was already the
whole job. The gateway derives routes, parameter schemas, validation, and
machine-readable docs from the signatures; pooling, isolation, retries,
and timeouts come from the decorator options. Any language can call in.

    app.serve(port=8283)                     # python
    fleetlet serve script.py --port 8283    # CLI ( --local = no VMs, dev)

Routes:
    GET  /                  service index: functions, schemas, docs
    GET  /health            {"ok": true}
    POST /call/<fn>         body = {params...} → wait, return result
    POST /spawn/<fn>        body = {params...} → {"call_id": ...}
    GET  /result/<call_id>  202 while running; result when done (handed out
                            once; unfetched results expire after ~15 min)

Request bodies are JSON parameter objects validated against the generated
schema (400 on mismatch). Results must be JSON-serializable; Python
clients may send `Accept: application/x-pickle` to receive arbitrary
objects pickled. Pickle is never ACCEPTED as input.

v0.1 has no auth and binds 127.0.0.1 by default — put a real proxy in
front of anything shared.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import pickle
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TYPE_CHECKING

from ._function import Function
from ._log import log
from ._schema import fn_schema, validate
from .errors import RemoteError, FleetletError, WorkerError

if TYPE_CHECKING:
    from .app import App

DEFAULT_PORT = 8283
MAX_BODY = 32 * 1024 * 1024
PICKLE_MIME = "application/x-pickle"
# Finished /spawn results a client never fetches are evicted after this long,
# so a fire-and-forget caller can't grow gateway memory without bound.
RESULT_TTL = 15 * 60.0


class Gateway:
    """Shared state behind the request handlers."""

    def __init__(self, app: "App", *, local: bool = False):
        self.app = app
        self.local = local
        self.functions: dict[str, Function] = dict(app._functions)
        self.schemas = {slug: fn_schema(fn._raw) for slug, fn in self.functions.items()}
        self.calls: dict[str, cf.Future] = {}
        self._done_at: dict[str, float] = {}  # call_id → completion time
        self._calls_lock = threading.Lock()
        self._executor = cf.ThreadPoolExecutor(max_workers=16) if local else None

    def submit(self, fn: Function, kwargs: dict[str, Any]) -> cf.Future:
        if self.local:
            return self._executor.submit(fn.local, **kwargs)
        return fn.spawn(**kwargs)

    def index(self) -> dict[str, Any]:
        functions = {}
        for slug, fn in self.functions.items():
            opts = fn._options
            functions[slug] = {
                "doc": (fn.__doc__ or "").strip() or None,
                "params": self.schemas[slug],
                "workers": 1 if opts.workers is None else opts.workers,
                "pool": opts.pool,
                "timeout": opts.timeout,
                "retries": opts.retries,
            }
        return {
            "app": self.app.name,
            "mode": "local" if self.local else "vm",
            "functions": functions,
            "endpoints": {
                "call": "POST /call/<fn>",
                "spawn": "POST /spawn/<fn>",
                "result": "GET /result/<call_id>",
                "health": "GET /health",
            },
        }

    def store_call(self, future: cf.Future) -> str:
        call_id = uuid.uuid4().hex[:12]
        with self._calls_lock:
            self._sweep_locked()
            self.calls[call_id] = future
        # Registered outside the lock: the callback fires synchronously when
        # the future is already done, and it takes the lock itself.
        future.add_done_callback(lambda _f, cid=call_id: self._mark_done(cid))
        return call_id

    def _mark_done(self, call_id: str) -> None:
        with self._calls_lock:
            if call_id in self.calls:
                self._done_at[call_id] = time.monotonic()

    def _sweep_locked(self) -> None:
        """Drop finished-but-unfetched results older than RESULT_TTL.
        Running calls are never evicted — only completed ones expire."""
        cutoff = time.monotonic() - RESULT_TTL
        for call_id, done in list(self._done_at.items()):
            if done < cutoff:
                self.calls.pop(call_id, None)
                self._done_at.pop(call_id, None)

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ routing

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        started = time.monotonic()
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/":
            self._reply(200, self.server.gateway.index(), started)
        elif path == "/health":
            self._reply(200, {"ok": True, "app": self.server.gateway.app.name}, started)
        elif path.startswith("/result/"):
            self._get_result(path.removeprefix("/result/"), started)
        else:
            self._not_found(path, started)

    def do_POST(self) -> None:  # noqa: N802
        started = time.monotonic()
        path = self.path.split("?", 1)[0].rstrip("/")
        if path.startswith("/call/"):
            self._invoke(path.removeprefix("/call/"), started, wait=True)
        elif path.startswith("/spawn/"):
            self._invoke(path.removeprefix("/spawn/"), started, wait=False)
        else:
            self._not_found(path, started)

    def _not_found(self, path: str, started: float) -> None:
        # POST /embed instead of /call/embed is everyone's first guess —
        # when the path names a known function, say so.
        slug = path.strip("/").rpartition("/")[2]
        hint = (f" — did you mean POST /call/{slug}?"
                if slug in self.server.gateway.functions else "")
        self._reply(404, {"ok": False, "error": _err(
            "NotFound", f"no route {path}{hint}")}, started)

    # ------------------------------------------------------------ handlers

    def _invoke(self, slug: str, started: float, *, wait: bool) -> None:
        gateway = self.server.gateway
        fn = gateway.functions.get(slug)
        if fn is None:
            self._reply(404, {"ok": False, "error": _err(
                "UnknownFunction", f"no function '{slug}' (see GET /)")}, started)
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._reply(400, {"ok": False, "error": _err("BadRequest", str(exc))}, started)
            return
        kwargs, errors = validate(gateway.schemas[slug], payload)
        if errors:
            self._reply(400, {"ok": False, "error": _err(
                "ValidationError", "; ".join(errors)), "errors": errors}, started)
            return

        try:
            future = gateway.submit(fn, kwargs)
        except FleetletError as exc:
            self._reply(503, {"ok": False, "error": _err(type(exc).__name__, str(exc))}, started)
            return

        if not wait:
            call_id = gateway.store_call(future)
            self._reply(202, {"ok": True, "call_id": call_id,
                              "result": f"/result/{call_id}"}, started)
            return
        self._reply_with_outcome(future, started, timeout=self._query_timeout())

    def _get_result(self, call_id: str, started: float) -> None:
        gateway = self.server.gateway
        with gateway._calls_lock:
            gateway._sweep_locked()
            future = gateway.calls.get(call_id)
        if future is None:
            self._reply(404, {"ok": False, "error": _err(
                "UnknownCall",
                f"no call '{call_id}' (results are handed out once, and "
                "expire unfetched after a while)")}, started)
            return
        if not future.done():
            self._reply(202, {"ok": True, "status": "running"}, started)
            return
        with gateway._calls_lock:
            gateway.calls.pop(call_id, None)
            gateway._done_at.pop(call_id, None)
        self._reply_with_outcome(future, started, timeout=0)

    def _reply_with_outcome(self, future: cf.Future, started: float,
                            timeout: float | None) -> None:
        try:
            result = future.result(timeout=timeout)
        except (cf.TimeoutError, TimeoutError) as exc:
            self._reply(504, {"ok": False, "error": _err(
                "Timeout", str(exc) or "call did not finish in time")}, started)
            return
        except RemoteError as exc:
            body = {"ok": False, "error": _err("RemoteError", exc.args[0] if exc.args else str(exc))}
            if getattr(exc, "remote_traceback", None):
                body["remote_traceback"] = exc.remote_traceback
            self._reply(500, body, started)
            return
        except WorkerError as exc:
            self._reply(502, {"ok": False, "error": _err("WorkerError", str(exc))}, started)
            return
        except Exception as exc:  # local-mode user exceptions land here
            self._reply(500, {"ok": False, "error": _err(type(exc).__name__, str(exc))}, started)
            return
        self._reply(200, {"ok": True, "result": result}, started)

    # ------------------------------------------------------------ plumbing

    def _read_json_body(self) -> Any:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"body exceeds {MAX_BODY // (1024 * 1024)}MB")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from None

    def _query_timeout(self) -> float | None:
        if "?" not in self.path:
            return None
        from urllib.parse import parse_qs
        qs = parse_qs(self.path.split("?", 1)[1])
        try:
            return float(qs["timeout"][0]) if "timeout" in qs else None
        except ValueError:
            return None

    def _reply(self, status: int, body: dict[str, Any], started: float) -> None:
        wants_pickle = PICKLE_MIME in (self.headers.get("Accept") or "")
        if wants_pickle:
            data, ctype = pickle.dumps(body), PICKLE_MIME
        else:
            try:
                data = json.dumps(body).encode()
            except (TypeError, ValueError):
                status = 500
                data = json.dumps({"ok": False, "error": _err(
                    "SerializationError",
                    "result is not JSON-serializable — request it with "
                    f"'Accept: {PICKLE_MIME}' from a python client")}).encode()
            ctype = "application/json"
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        log(f"{self.command} {self.path} -> {status} "
            f"({(time.monotonic() - started) * 1000:.0f}ms)")

    def log_message(self, *args: Any) -> None:  # silence stdlib per-request lines
        pass


def _err(etype: str, message: str) -> dict[str, str]:
    return {"type": etype, "message": message}


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], gateway: Gateway):
        super().__init__(address, _Handler)
        self.gateway = gateway


def make_server(app: "App", host: str = "127.0.0.1", port: int = DEFAULT_PORT,
                *, local: bool = False) -> _Server:
    """Build (but don't run) the HTTP server — the unit-testable seam."""
    if not app._functions:
        raise FleetletError("app has no @app.function definitions to serve")
    return _Server((host, port), Gateway(app, local=local))


def serve(app: "App", host: str = "127.0.0.1", port: int = DEFAULT_PORT,
          *, warm: bool = False, local: bool = False) -> None:
    """Serve until interrupted. VM mode runs inside `app.run()` (pools boot
    lazily on first call, or all up-front with warm=True); local mode runs
    functions in-process — instant startup, no isolation, dev only."""
    server = make_server(app, host, port, local=local)
    bound = f"http://{server.server_address[0]}:{server.server_address[1]}"
    names = ", ".join(server.gateway.functions)

    # A VM-backed service MUST tear down on SIGTERM (docker stop, k8s,
    # systemd) or it leaks machines. Raising KeyboardInterrupt reuses the
    # ^C path: it unwinds serve_forever and exits `with app.run()` cleanly.
    # Only possible from the main thread; elsewhere, rely on the caller.
    def _sigterm_to_interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    previous_sigterm = None
    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.signal(signal.SIGTERM, _sigterm_to_interrupt)
    try:
        if local:
            log(f"serving '{app.name}' at {bound} (LOCAL mode — no VMs): {names}")
            server.serve_forever()
        else:
            with app.run():
                if warm:
                    for fn in server.gateway.functions.values():
                        app._pool_for_function(fn).ensure_started()
                log(f"serving '{app.name}' at {bound}: {names}")
                server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        server.shutdown()
        server.server_close()
        server.gateway.close()
