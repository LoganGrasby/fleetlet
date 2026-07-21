"""Worker pools: fleets of smolvm microVMs behind a task queue.

Two strategies:

* ``cold`` — every worker is an independent machine: create → start → stage
  files → bake image steps → launch runner → setup. Each worker pays the
  full boot + bake cost.

* ``fork`` — ONE golden machine pays that cost, gets warmed (imports done,
  actor constructed, @enter hooks run), then is frozen and cloned with
  `machine fork` in ~100ms per worker. Clones resume with the runner already
  listening and all state in memory — smolvm's answer to Modal's memory
  snapshots. Scale-up later keeps forking the frozen golden.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import hashlib
import json
import pickle
import queue
import secrets
import select
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import cloudpickle

from . import _machine
from ._backend import Backend, get_backend
from ._log import log, log_remote_output
from ._proto import (
    GUEST_LOG,
    GUEST_PORT,
    GUEST_PROJECT,
    GUEST_ROOT,
    GUEST_RUNNER,
    PICKLE_PROTOCOL,
    recv_frame,
    send_frame,
)
from .errors import CloudError, ConfigError, RemoteError, SmolvmError, WorkerError
from .image import Image

CONTROL_TIMEOUT = 60.0
COLD_CONNECT_DEADLINE = 120.0
FORK_CONNECT_DEADLINE = 30.0
READ_TIMEOUT = 60.0          # max gap between bytes once a reply starts arriving
LIVENESS_POLL = 5.0          # how often to probe a quiet worker for death
LIVENESS_PROBE_TIMEOUT = 20.0
CLOUD_CONNECT_DEADLINE = 240.0   # create + start + ingress DNS can take a while
CLOUD_FORK_CONNECT_DEADLINE = 120.0  # forks serialize per golden + exec cold-start
HTTP_POLL_WAIT = 20.0        # server-side long-poll hold per /poll request

# In-guest bridge for cloud fork clones: reads one base64 pickle frame from
# stdin, speaks the runner's length-prefixed socket protocol on 127.0.0.1
# (clones keep the golden's processes, and execs share its network
# namespace), prints the reply base64 on stdout.
# Runs inside `python3 -c` via the exec API — no ingress, no open ports.
RELAY_SRC = """
import base64, socket, struct, sys
data = base64.b64decode(sys.stdin.buffer.read())
s = socket.create_connection(("127.0.0.1", {port}), 15)
s.sendall(struct.pack(">I", len(data)) + data)
def rx(n):
    buf = b""
    while len(buf) < n:
        c = s.recv(n - len(buf))
        if not c:
            print("RELAY-EOF", file=sys.stderr); raise SystemExit(3)
        buf += c
    return buf
(ln,) = struct.unpack(">I", rx(4))
sys.stdout.write("B64:" + base64.b64encode(rx(ln)).decode())
"""

_used_ports: set[int] = set()
_port_lock = threading.Lock()


class _timed:
    """Context manager logging a bringup phase's duration (debug-level UX)."""

    def __init__(self, name: str, phase: str):
        self.label = f"{name}: {phase}"

    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        import os

        if os.environ.get("FLEETLET_TIMINGS"):
            log(f"{self.label} [{time.monotonic() - self.t0:.2f}s]")
        return False


def alloc_port() -> int:
    """Pick a free host port for a worker's guest-port forward."""
    with _port_lock:
        for _ in range(64):
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            s.close()
            if port not in _used_ports:
                _used_ports.add(port)
                return port
    raise WorkerError("could not allocate a free host port")


@dataclass
class PoolConfig:
    app_name: str
    fn_slug: str
    run_id: str
    image: Image
    mode: str = "auto"  # auto | fork | cold
    size: int = 1
    cpus: int = 2
    memory_mib: int = 1024
    gpu: bool = False
    gpu_vram_mib: int | None = None
    cuda: bool = False
    share_weights: bool = False
    net: bool | list[str] = False
    volumes: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    project_tar: str | None = None
    main_file_rel: str | None = None
    stage_tar: str | None = None  # runner + cloudpickle + fleetlet pkg
    cls_spec: dict[str, Any] | None = None
    cls_blob: bytes | None = None
    cls_args: bytes | None = None  # cloudpickle((args, kwargs))
    setup_id: str = ""
    call_timeout: float | None = None
    retries: int = 0

    def resolved_mode(self) -> str:
        if self.mode != "auto":
            return self.mode
        return "fork" if self.size > 1 else "cold"

    def machine_prefix(self) -> str:
        # The readable segments are truncated, so two long slugs can share
        # them; the digest keeps prefixes unique — teardown sweeps machines
        # by prefix, so a collision would delete another pool's workers.
        ident = hashlib.sha256(
            f"{self.app_name}/{self.fn_slug}".encode()
        ).hexdigest()[:6]
        return f"flt-{self.app_name[:12]}-{self.fn_slug[:12]}-{ident}-{self.run_id}"

    def validate(self) -> None:
        if self.image.needs_network and self.net is not True:
            raise ConfigError(
                f"image for '{self.fn_slug}' has build steps (pip/apt/run_commands), "
                "which run inside the guest and need egress. Set net=True on the "
                "function, or use a pre-built registry image with no steps."
            )
        if self.resolved_mode() == "fork" and self.volumes:
            log(f"warning: fork pool '{self.fn_slug}' uses volumes — volume mounts "
                "on fork clones are untested; prefer cold pools for volume workloads")


@dataclass
class Task:
    spec: dict[str, Any]
    blob: bytes | None
    args_blob: bytes
    timeout: float | None
    retries: int
    tag: str
    future: cf.Future = field(default_factory=cf.Future)


_SENTINEL = Task(spec={}, blob=None, args_blob=b"", timeout=None, retries=0, tag="")


class Worker:
    """Host-side handle to one machine's runner. Transport-agnostic ops here;
    `SocketWorker` (local TCP forward) and `HttpWorker` (cloud ingress)
    provide wait_ready/rpc/liveness/close."""

    def __init__(self, name: str):
        self.name = name
        self.sent_blobs: set[str] = set()
        self.guest_py: tuple[int, ...] = ()
        self.has_cloudpickle = False

    def wait_ready(self, deadline: float) -> None:
        raise NotImplementedError

    def rpc(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        raise NotImplementedError

    def is_runner_alive(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        pass

    # ------------------------------------------------------------ ops

    def hello(self) -> None:
        resp = self.rpc({"op": "hello"}, CONTROL_TIMEOUT)
        self.guest_py = tuple(resp.get("py", ()))
        self.has_cloudpickle = bool(resp.get("cloudpickle"))

    def setup(self, cfg: PoolConfig) -> None:
        resp = self.rpc(
            {
                "op": "setup",
                "setup_id": cfg.setup_id,
                "sys_path": [GUEST_ROOT, GUEST_PROJECT],
                "main_file": cfg.main_file_rel,
                "cls_spec": cfg.cls_spec,
                "cls_blob": cfg.cls_blob,
                "cls_args": cfg.cls_args or cloudpickle.dumps(((), {})),
            },
            None,  # @enter hooks may load models — no deadline
        )
        if not resp.get("ok"):
            raise _remote_error(resp["error"], f"worker {self.name} setup failed")

    def call(self, task: Task) -> Any:
        payload: dict[str, Any] = {
            "op": "call",
            "spec": task.spec,
            "args": task.args_blob,
        }
        needs_blob = (
            task.spec.get("kind") == "blob"
            and task.spec["blob_id"] not in self.sent_blobs
        )
        if needs_blob:
            payload["blob"] = task.blob
        resp = self.rpc(payload, task.timeout)
        if resp.get("need_blob"):
            payload["blob"] = task.blob
            resp = self.rpc(payload, task.timeout)
        if task.spec.get("kind") == "blob":
            self.sent_blobs.add(task.spec["blob_id"])

        if resp.get("stdout") or resp.get("stderr"):
            log_remote_output(task.tag, resp.get("stdout", ""), resp.get("stderr", ""))
        if resp.get("ok"):
            return pickle.loads(resp["value"])
        raise _remote_error(resp["error"], f"{task.tag} raised remotely")


class SocketWorker(Worker):
    """Local target: raw TCP frames over the machine's forwarded host port."""

    def __init__(self, name: str, port: int, backend: Backend):
        super().__init__(name)
        self.port = port
        self.backend = backend
        self.sock: socket.socket | None = None

    def wait_ready(self, deadline: float) -> None:
        """Establish a live connection to the runner.

        smolvm's TSI port-forward accepts the host-side TCP connection even
        before the guest runner is listening, then resets on first read. So a
        bare connect() is not a readiness signal — we retry a full `hello`
        round-trip, reconnecting through resets/EOF, until one succeeds.
        """
        end = time.monotonic() + deadline
        last_err: Exception | None = None
        while time.monotonic() < end:
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sock = sock
                self.hello()
                return
            except (OSError, WorkerError) as exc:
                last_err = exc
                self.close()
                time.sleep(0.15)
        raise WorkerError(
            f"worker {self.name}: runner not reachable on 127.0.0.1:{self.port} "
            f"after {deadline:.0f}s ({last_err!r}). Guest-side log: "
            f"`smolvm machine exec --name {self.name} -- cat {GUEST_LOG}`"
        )

    def rpc(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        """Send a request, wait for the reply.

        smolvm's TSI forward does NOT propagate a guest-side socket close to
        the host — if the runner dies mid-call, our recv would block forever
        (no EOF, no reset). So instead of a blocking recv we poll the socket
        with `select`, and whenever it goes quiet for LIVENESS_POLL seconds we
        actively probe whether the runner is still alive. A slow-but-alive
        call keeps waiting; a dead worker fails fast.
        """
        if self.sock is None:
            raise WorkerError(f"worker {self.name} is not connected")
        try:
            send_frame(self.sock, payload)
        except OSError as exc:
            raise WorkerError(f"worker {self.name} send failed: {exc!r}") from exc

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"call on worker {self.name} exceeded {timeout}s")
                wait = min(LIVENESS_POLL, remaining)
            else:
                wait = LIVENESS_POLL
            readable, _, _ = select.select([self.sock], [], [], wait)
            if readable:
                return self._recv_reply()
            if not self.is_runner_alive():
                raise WorkerError(
                    f"worker {self.name}: runner died mid-call "
                    "(process exited, VM crashed, or OOM-killed)"
                )

    def _recv_reply(self) -> dict[str, Any]:
        try:
            self.sock.settimeout(READ_TIMEOUT)
            resp = recv_frame(self.sock)
        except socket.timeout:
            raise WorkerError(f"worker {self.name} stalled mid-frame") from None
        except OSError as exc:
            raise WorkerError(f"worker {self.name} connection failed: {exc!r}") from exc
        if resp is None:
            raise WorkerError(f"worker {self.name} closed the connection")
        return resp

    def is_runner_alive(self) -> bool:
        return _socket_liveness_probe(self.backend, self.name)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def _socket_liveness_probe(backend: Backend, name: str) -> bool:
    """Active liveness probe via a side-channel exec.

    Asks the guest kernel about the runner's listen socket: a connect to
    it from inside the guest lands in the accept backlog while the runner
    lives (even mid-call), and is REFUSED once it dies. PID probes can't
    work here — each exec runs in its own PID namespace, where the
    detach-reported (guest-global) runner pid doesn't exist.

    Only two answers are trusted: a clean refusal means dead, VM state
    not-running means dead. Everything else — exec noise, probe timeout,
    full backlog — is benefit of the doubt: false-killing a busy worker
    costs a warm replacement; waiting costs one more poll.
    """
    st = backend.status(name)
    if st is None or st.get("state") != "running":
        return False
    snippet = (
        "import socket\n"
        "try:\n"
        f"    socket.create_connection(('127.0.0.1',{GUEST_PORT}),3).close();print('A')\n"
        "except ConnectionRefusedError:\n"
        "    print('D')\n"
        "except Exception:\n"
        "    print('A')\n"
    )
    probe = backend.execute(
        name, ["python3", "-c", snippet],
        check=False, timeout=LIVENESS_PROBE_TIMEOUT,
    )
    if probe.returncode != 0:
        return True  # exec-side noise (e.g. agent settling) — don't false-kill
    return b"D" not in probe.stdout


class HttpWorker(Worker):
    """Cloud target: ops as pickle bodies over the machine's ingress URL.

    Two auth layers ride every request: the platform's tenant check on the
    ingress (Bearer API key) and the runner's own per-run token. Long calls
    use the runner's submit/poll pair so no single HTTP request outlives a
    proxy timeout. Results are retained guest-side until the next submit,
    so one dropped poll response doesn't lose a finished call.
    """

    def __init__(self, name: str, backend: Any, token: str):
        super().__init__(name)
        self.backend = backend
        self.token = token
        self.base_url: str | None = None

    # ------------------------------------------------------------ http plumbing

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.backend.api_key}",
            "x-fleetlet-token": self.token,
        }

    def _http(self, method: str, path: str, body: bytes | None = None,
              timeout: float = 30.0) -> tuple[int, bytes]:
        req = urllib.request.Request(
            self.base_url + path, data=body, headers=self._headers(), method=method,
        )
        if body is not None:
            req.add_header("content-type", "application/octet-stream")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    # ------------------------------------------------------------ transport

    def wait_ready(self, deadline: float) -> None:
        end = time.monotonic() + deadline
        self.base_url = self.backend.ingress_url(
            self.name, deadline_s=max(end - time.monotonic(), 10.0)
        ).rstrip("/")
        last: Any = None
        while time.monotonic() < end:
            try:
                status, body = self._http("GET", "/healthz", timeout=10)
                if status == 200 and body == b"ok":
                    self.hello()
                    return
                last = (status, body[:120])
            except (OSError, urllib.error.URLError) as exc:
                last = exc
            time.sleep(1.0)
        raise WorkerError(
            f"worker {self.name}: runner not reachable at {self.base_url} "
            f"after {deadline:.0f}s (last: {last!r})"
        )

    def rpc(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        body = pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
        status, resp = self._submit(body)
        if status == 403:
            raise WorkerError(f"worker {self.name}: runner rejected our token")
        if status != 202:
            raise WorkerError(
                f"worker {self.name}: submit failed ({status}: {resp[:160]!r})"
            )
        job_id = json.loads(resp)["id"]

        deadline = None if timeout is None else time.monotonic() + timeout
        misses = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"call on worker {self.name} exceeded {timeout}s")
            try:
                status, resp = self._http(
                    "GET", f"/poll?id={job_id}&wait={HTTP_POLL_WAIT:.0f}",
                    timeout=HTTP_POLL_WAIT + 30,
                )
                misses = 0
            except (OSError, urllib.error.URLError) as exc:
                # Transient ingress blips shouldn't kill a running call; a
                # dead VM/runner shouldn't stall it. Probe, then retry a few.
                misses += 1
                if not self.is_runner_alive():
                    raise WorkerError(
                        f"worker {self.name}: runner died mid-call ({exc!r})"
                    ) from None
                if misses >= 5:
                    raise WorkerError(
                        f"worker {self.name}: lost contact mid-call ({exc!r})"
                    ) from None
                time.sleep(1.0)
                continue
            if status == 200:
                return pickle.loads(resp)
            if status == 204:
                continue
            raise WorkerError(
                f"worker {self.name}: poll failed ({status}: {resp[:160]!r})"
            )

    def _submit(self, body: bytes) -> tuple[int, bytes]:
        try:
            return self._http("POST", "/submit", body, timeout=60.0)
        except (OSError, urllib.error.URLError) as exc:
            raise WorkerError(f"worker {self.name}: submit failed ({exc!r})") from None

    def is_runner_alive(self) -> bool:
        st = self.backend.status(self.name)
        if st is None or st.get("state") != "running":
            return False
        try:
            status, body = self._http("GET", "/healthz", timeout=10)
            return status == 200
        except (OSError, urllib.error.URLError):
            return True  # ingress blip ≠ dead runner; benefit of the doubt


class ExecRelayWorker(Worker):
    """Cloud fork clones: ops bridged to the INHERITED socket runner through
    the exec API (stdin → 127.0.0.1 → stdout, base64-framed).

    Clones keep the golden's running processes but get no ingress URL, so
    HttpWorker can't reach them; execs can (shared guest network namespace).
    Each op is one exec whose lifetime IS the call — no separate liveness
    polling loop needed mid-call. Everything is authenticated at the cloud
    API; the machine exposes no reachable port at all.

    Trade-offs vs HttpWorker: ~exec-RTT per op, and base64 through the exec
    endpoint caps single results at roughly 14MB (stdout limit). Untimed ops
    are bounded by the exec API's hard cap (EXEC_TIMEOUT, 1800s) — a
    synchronous exec cannot run unbounded.
    """

    def __init__(self, name: str, backend: Any):
        super().__init__(name)
        self.backend = backend
        self._relay_src = RELAY_SRC.format(port=GUEST_PORT)

    def wait_ready(self, deadline: float) -> None:
        end = time.monotonic() + deadline
        last: Exception | None = None
        while time.monotonic() < end:
            try:
                self.hello()
                return
            except (WorkerError, TimeoutError) as exc:
                last = exc
                time.sleep(1.0)
        raise WorkerError(
            f"worker {self.name}: inherited runner not answering the exec "
            f"relay after {deadline:.0f}s ({last!r})"
        )

    def rpc(self, payload: dict[str, Any], timeout: float | None) -> dict[str, Any]:
        blob = base64.b64encode(
            pickle.dumps(payload, protocol=PICKLE_PROTOCOL)
        ).decode()
        machine_id = self.backend._require_id(self.name)
        exec_timeout = timeout if timeout is not None else _machine.EXEC_TIMEOUT
        try:
            resp = self.backend.client.exec(
                machine_id, ["python3", "-c", self._relay_src],
                stdin=blob, timeout=exec_timeout,
            )
        except CloudError as exc:
            raise WorkerError(f"worker {self.name}: relay exec failed ({exc})") from None
        out = str(resp.get("stdout", ""))
        if resp.get("exitCode", 1) != 0 or "B64:" not in out:
            if "timed out" in str(resp.get("stderr", "")).lower():
                msg = f"call on worker {self.name} exceeded {exec_timeout:.0f}s"
                if timeout is None:
                    msg += (" — the exec relay bounds untimed ops at "
                            "EXEC_TIMEOUT; pass an explicit timeout for longer calls")
                raise TimeoutError(msg)
            raise WorkerError(
                f"worker {self.name}: relay rc={resp.get('exitCode')} "
                f"stderr={str(resp.get('stderr', ''))[-160:]!r}"
            )
        if resp.get("stdoutTruncated"):
            raise WorkerError(
                f"worker {self.name}: result too large for the exec relay "
                "(~14MB cap) — return smaller values from fork-pool functions"
            )
        return pickle.loads(base64.b64decode(out.split("B64:", 1)[1]))

    def is_runner_alive(self) -> bool:
        return _socket_liveness_probe(self.backend, self.name)


def _complete(fut_method, *args: Any) -> None:
    """Deliver a future outcome, tolerating a racing Future.cancel() — an
    InvalidStateError here must never kill a pool's serve thread."""
    try:
        fut_method(*args)
    except cf.InvalidStateError:
        pass  # future was cancelled; nothing to deliver


def _remote_error(err: dict[str, Any], context: str) -> BaseException:
    """Rebuild the guest exception if transportable, else a RemoteError."""
    remote_tb = err.get("traceback", "")
    pickled = err.get("pickled")
    if pickled:
        try:
            exc = pickle.loads(pickled)
        except Exception:
            exc = None
        if isinstance(exc, BaseException):
            if hasattr(exc, "add_note"):
                exc.add_note(f"(raised in {context})\n--- remote traceback ---\n{remote_tb.rstrip()}")
            return exc
    return RemoteError(f"{context}: {err.get('etype')}: {err.get('repr')}", remote_tb)


class Pool:
    def __init__(self, cfg: PoolConfig, backend: Backend | None = None):
        cfg.validate()
        self.cfg = cfg
        self.backend = backend or get_backend()
        self.mode = cfg.resolved_mode()
        if self.backend.target == "cloud" and self.mode == "fork":
            # Cloud clones inherit the golden's warm runner but get no
            # ingress URL, so their calls ride the exec API.
            log(f"pool '{cfg.fn_slug}': cloud fork pool — clones keep the "
                "golden's warm state; calls go over the exec relay")
        # Second auth layer for cloud workers, on top of the tenant-authed
        # ingress: a per-pool secret the runner requires on every op request.
        self._runner_token = secrets.token_hex(16)
        self.golden: str | None = None
        self.workers: list[Worker] = []
        self.threads: list[threading.Thread] = []
        self.tasks: "queue.SimpleQueue[Task]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._worker_seq = 0
        self._started = False
        self._closed = False
        self._py_warned = False

    # ------------------------------------------------------------ lifecycle

    def ensure_started(self, size: int | None = None) -> None:
        with self._lock:
            target = max(self.cfg.size, size or 0, 1)
            if not self._started:
                self._started = True
                started_at = time.monotonic()
                if self.mode == "fork":
                    self._bringup_golden()
                self._add_workers_locked(target)
                log(f"pool '{self.cfg.fn_slug}' ready: {len(self.workers)} "
                    f"{self.mode} worker(s) in {time.monotonic() - started_at:.1f}s")
            elif target > len(self.workers):
                self._add_workers_locked(target - len(self.workers))

    def _add_workers_locked(self, count: int) -> None:
        new_names: list[str] = []
        for _ in range(count):
            idx = self._worker_seq
            self._worker_seq += 1
            new_names.append(f"{self.cfg.machine_prefix()}-w{idx}")
        with cf.ThreadPoolExecutor(max_workers=max(len(new_names), 1)) as pool:
            futures = [pool.submit(self._bringup_worker, name) for name in new_names]
            workers, failures = [], []
            for f in futures:
                try:
                    workers.append(f.result())
                except Exception as exc:
                    failures.append(exc)
        # Register every worker that DID come up, even when siblings failed —
        # otherwise those live machines would sit outside the pool (no serve
        # thread, invisible to scaling) until the teardown sweep.
        for worker in workers:
            self.workers.append(worker)
            thread = threading.Thread(
                target=self._serve, args=(worker,),
                name=f"fleetlet-{worker.name}", daemon=True,
            )
            self.threads.append(thread)
            thread.start()
        if failures:
            for extra in failures[1:]:
                log(f"additional worker bringup failure: {extra}")
            raise failures[0]

    # ------------------------------------------------------------ bringup

    def _machine_spec(self, name: str, port: int) -> _machine.MachineSpec:
        cfg = self.cfg
        return _machine.MachineSpec(
            name=name,
            image=cfg.image.base,
            cpus=cfg.cpus,
            memory_mib=cfg.memory_mib,
            # Local: forward a host port to the runner. Cloud: publish the
            # guest port; the control plane allocates the host side and
            # routes the ingress URL to it.
            ports=[(port, GUEST_PORT)],
            volumes=list(cfg.volumes),
            net=cfg.net is True,
            allow_hosts=cfg.net if isinstance(cfg.net, list) else [],
            gpu=cfg.gpu,
            gpu_vram_mib=cfg.gpu_vram_mib,
            cuda=cfg.cuda,
        )

    def _runner_env(self) -> dict[str, str]:
        env = dict(self.cfg.image.env_vars)
        env.update(self.cfg.env)
        if self.cfg.cuda and self.cfg.share_weights:
            # Weights must land in VMM chunks for cross-clone sharing.
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env

    def _stage_and_launch(self, name: str, *, runner_http: bool | None = None) -> None:
        """Copy runner + cloudpickle + project into the guest, bake image
        steps, and launch the runner server detached."""
        cfg = self.cfg
        backend = self.backend
        if runner_http is None:
            runner_http = backend.target == "cloud" and self.mode != "fork"
        if not _probe_python3(backend, name):
            raise ConfigError(
                f"image '{cfg.image.base}' has no python3 — fleetlet workers need a "
                "Python image (e.g. Image.default() or python:3.12-slim)"
            )

        with _timed(name, "stage runtime"):
            backend.execute(name, ["sh", "-c", f"mkdir -p {GUEST_ROOT} {GUEST_PROJECT}"])
            stage = cfg.stage_tar or build_stage_tar()
            backend.put_file(name, stage, f"{GUEST_ROOT}/stage.tgz")
            backend.execute(name, ["sh", "-c",
                                   f"tar -xzf {GUEST_ROOT}/stage.tgz -C {GUEST_ROOT} "
                                   f"&& rm {GUEST_ROOT}/stage.tgz"])
        if cfg.project_tar:
            # Not /tmp: `machine cp` and `machine exec` can see different
            # tmpfs namespaces; GUEST_ROOT is on the shared overlay disk.
            with _timed(name, "sync project"):
                tar_path = f"{GUEST_ROOT}/project.tgz"
                backend.put_file(name, cfg.project_tar, tar_path)
                backend.execute(name, ["sh", "-c",
                                       f"tar -xzf {tar_path} -C {GUEST_PROJECT} "
                                       f"&& rm {tar_path}"])

        env = self._runner_env()
        for i, step in enumerate(cfg.image.steps):
            with _timed(name, f"bake {i + 1}/{len(cfg.image.steps)}: {step[:50]}"):
                backend.execute(name, ["sh", "-c", step], env=env)

        workdir = cfg.image.workdir_path or (GUEST_PROJECT if cfg.project_tar else GUEST_ROOT)
        runner_cmd = ["python3", GUEST_RUNNER, "--port", str(GUEST_PORT),
                      "--project-root", GUEST_PROJECT]
        # Cloud COLD workers are reached over the ingress → HTTP runner.
        # Cloud FORK goldens keep the socket runner: clones have no ingress,
        # so their traffic rides the exec relay straight to the socket.
        if runner_http:
            runner_cmd.append("--http")
            env = {**env, "FLEETLET_RUNNER_TOKEN": self._runner_token}
        with _timed(name, "launch runner"):
            backend.execute(name, runner_cmd, env=env, workdir=workdir, detach=True)

    def _finish_worker(self, worker: Worker, deadline: float) -> Worker:
        worker.wait_ready(deadline)
        self._check_guest_python(worker)
        worker.setup(self.cfg)
        return worker

    def _check_guest_python(self, worker: Worker) -> None:
        host = sys.version_info[:2]
        if worker.guest_py and tuple(worker.guest_py[:2]) != host and not self._py_warned:
            self._py_warned = True
            log(f"warning: guest python {worker.guest_py[0]}.{worker.guest_py[1]} != "
                f"host {host[0]}.{host[1]} — module-level functions are fine, but "
                "lambdas/closures ship as bytecode and may fail. Match versions with "
                "Image.default().")

    def _bringup_golden(self) -> None:
        cfg = self.cfg
        cloud = self.backend.target == "cloud"
        name = f"{cfg.machine_prefix()}-g"
        port = 0 if cloud else alloc_port()
        log(f"building fork golden {name} ({cfg.image.base})…")
        t0 = time.monotonic()
        spec = self._machine_spec(name, port)
        spec.forkable = True  # cloud: create-body flag; local: no-op field
        with _timed(name, "create"):
            self.backend.create(spec)
        with _timed(name, "start --forkable"):
            self.backend.start(name, forkable=True)
        self._stage_and_launch(name)
        golden_conn: Worker = (
            ExecRelayWorker(name, self.backend) if cloud
            else SocketWorker(name, port, self.backend)
        )
        with _timed(name, "ready+setup"):
            self._finish_worker(golden_conn, COLD_CONNECT_DEADLINE)
        # Freeze point: the golden must be idle in accept() when forked.
        golden_conn.close()
        self.golden = name
        log(f"golden {name} warm in {time.monotonic() - t0:.1f}s "
            "(imports + @enter done; clones inherit this state)")

    def _bringup_worker(self, name: str) -> Worker:
        t0 = time.monotonic()
        if self.mode == "fork" and self.backend.target == "cloud":
            try:
                self.backend.fork(self.golden, name, ports=[(0, GUEST_PORT)],
                                  share_weights=False)
            except (SmolvmError, CloudError) as exc:
                # e.g. the golden landed on a node without a fork-capable
                # engine, or its fork control socket has gone away. One cold
                # worker beats a dead pool.
                log(f"fork of {name} failed ({exc}); provisioning cold instead")
                return self._bringup_cold_cloud_worker(name, t0)
            worker = self._finish_worker(
                ExecRelayWorker(name, self.backend), CLOUD_FORK_CONNECT_DEADLINE)
            log(f"forked {name} in {time.monotonic() - t0:.2f}s (warm clone)")
        elif self.mode == "fork":
            port = alloc_port()
            self.backend.fork(
                self.golden, name,
                ports=[(port, GUEST_PORT)],
                share_weights=self.cfg.share_weights,
            )
            worker = self._finish_worker(
                SocketWorker(name, port, self.backend), FORK_CONNECT_DEADLINE)
            log(f"forked {name} in {time.monotonic() - t0:.2f}s")
        elif self.backend.target == "cloud":
            worker = self._bringup_cold_cloud_worker(name, t0)
        else:
            port = alloc_port()
            self.backend.create(self._machine_spec(name, port))
            self.backend.start(name)
            self._stage_and_launch(name)
            worker = self._finish_worker(
                SocketWorker(name, port, self.backend), COLD_CONNECT_DEADLINE)
            log(f"cold worker {name} up in {time.monotonic() - t0:.1f}s")
        return worker

    def _bringup_cold_cloud_worker(self, name: str, t0: float) -> Worker:
        self.backend.create(self._machine_spec(name, 0))
        self.backend.start(name)
        self._stage_and_launch(name, runner_http=True)
        worker = self._finish_worker(
            HttpWorker(name, self.backend, self._runner_token),
            CLOUD_CONNECT_DEADLINE)
        log(f"cloud worker {name} up in {time.monotonic() - t0:.1f}s")
        return worker

    def _replace_worker(self, dead: Worker) -> Worker:
        dead.close()
        self.backend.stop(dead.name)
        self.backend.delete(dead.name)
        with self._lock:
            idx = self._worker_seq
            self._worker_seq += 1
        name = f"{self.cfg.machine_prefix()}-w{idx}"
        replacement = self._bringup_worker(name)
        with self._lock:
            if dead in self.workers:
                self.workers[self.workers.index(dead)] = replacement
            else:
                self.workers.append(replacement)
        return replacement

    # ------------------------------------------------------------ dispatch

    def submit(self, spec: dict[str, Any], blob: bytes | None,
               args: tuple, kwargs: dict, *, tag: str,
               timeout: float | None = None, size_hint: int | None = None) -> cf.Future:
        self.ensure_started(size_hint)
        task = Task(
            spec=spec,
            blob=blob,
            args_blob=cloudpickle.dumps((args, kwargs)),
            timeout=timeout if timeout is not None else self.cfg.call_timeout,
            retries=self.cfg.retries,
            tag=tag,
        )
        self.tasks.put(task)
        return task.future

    def _serve(self, worker: Worker) -> None:
        while True:
            task = self.tasks.get()
            if task is _SENTINEL:
                return
            # Mark RUNNING so a user cancel() can no longer race the delivery
            # below. A re-queued retry task is already RUNNING — skip it.
            if not task.future.running():
                if not task.future.set_running_or_notify_cancel():
                    continue  # cancelled while queued
            try:
                result = worker.call(task)
            except (WorkerError, TimeoutError) as exc:
                worker = self._handle_transport_failure(worker, task, exc)
                continue
            except BaseException as exc:  # remote user exception
                _complete(task.future.set_exception, exc)
                continue
            _complete(task.future.set_result, result)

    def _handle_transport_failure(
        self, worker: Worker, task: Task, exc: Exception
    ) -> Worker:
        """The worker VM (or its connection) died mid-call. Replace it —
        cheap for fork pools — and retry the task if allowed."""
        if self._closed:
            # The transport may be desynced (a timed-out call's reply is still
            # owed on the stream, and replies match requests by order alone) —
            # drop it so no queued task can read a stale frame as its answer.
            worker.close()
            _complete(task.future.set_exception, exc)
            return worker
        log(f"{worker.name} failed ({exc}); replacing")
        try:
            replacement = self._replace_worker(worker)
        except Exception as bringup_exc:
            _complete(task.future.set_exception,
                      WorkerError(f"worker replacement failed: {bringup_exc}"))
            # Prune this worker and its serve thread before the re-raise kills
            # the thread, so the next submit()'s ensure_started sees the lost
            # capacity and rebuilds it.
            with self._lock:
                if worker in self.workers:
                    self.workers.remove(worker)
                current = threading.current_thread()
                if current in self.threads:
                    self.threads.remove(current)
                pool_empty = not self.workers
            if pool_empty:
                # That was the last serve thread. A caller already blocked on
                # queued futures can never trigger the rebuild, so fail those
                # tasks now instead of stranding them PENDING forever.
                self._fail_queued_tasks(
                    f"worker replacement failed: {bringup_exc}")
            raise
        if task.retries > 0 and not self._closed:
            # The _closed re-check matters: shutdown() may have run during the
            # slow replacement, and a task requeued behind its sentinels would
            # never be served (and its RUNNING future can't be cancelled).
            task.retries -= 1
            self.tasks.put(task)
        else:
            _complete(task.future.set_exception, exc)
        return replacement

    def _fail_queued_tasks(self, reason: str) -> None:
        """Fail every task still in the queue — used when the pool loses its
        last serve thread, or shuts down with work outstanding. Sentinels are
        re-queued for any threads still draining."""
        sentinels = 0
        while True:
            try:
                task = self.tasks.get_nowait()
            except queue.Empty:
                break
            if task is _SENTINEL:
                sentinels += 1
                continue
            _complete(task.future.set_exception, WorkerError(reason))
        for _ in range(sentinels):
            self.tasks.put(_SENTINEL)

    # ------------------------------------------------------------ teardown

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            workers = list(self.workers)
        for _ in self.threads:
            self.tasks.put(_SENTINEL)
        for thread in self.threads:
            thread.join(timeout=5)
        # Fail anything still queued: no consumer will ever run it now, and a
        # waiter cannot cancel a future that has already been marked RUNNING.
        self._fail_queued_tasks(
            f"pool '{self.cfg.fn_slug}' shut down before this call ran")
        for worker in workers:
            worker.close()
        # Sweep by name prefix so machines from half-finished bringups are
        # cleaned too, not only the ones that made it into self.workers.
        prefix = self.cfg.machine_prefix()
        names = {w.name for w in workers}
        try:
            names.update(
                m["name"] for m in self.backend.list_machines()
                if m["name"].startswith(prefix)
            )
        except (SmolvmError, CloudError) as exc:
            log(f"machine listing failed during teardown ({exc}); "
                "cleaning only the workers this pool tracked")
        golden = self.golden or f"{prefix}-g"
        clones = sorted(names - {golden})
        if clones or golden in names or self.golden:
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(self._teardown_machine, clones))
            self._teardown_machine(golden)
            log(f"pool '{self.cfg.fn_slug}' torn down "
                f"({len(clones)} worker(s){' + golden' if self.golden else ''})")

    def _teardown_machine(self, name: str) -> None:
        try:
            self.backend.stop(name)
            if not self.backend.delete(name):
                # stop/delete report failure as False, not an exception —
                # say so instead of letting the machine leak silently.
                log(f"cleanup of {name} incomplete — run `fleetlet clean`")
        except SmolvmError as exc:  # pragma: no cover — best-effort cleanup
            log(f"cleanup of {name} failed: {exc}")


# ---------------------------------------------------------------- staging helpers

def _probe_python3(backend: Backend, name: str, deadline_s: float = 20.0) -> bool:
    """True once the guest has python3 on PATH. Polls: right after `start`
    (notably on packed `--from` machines) execs can get killed while the
    agent settles (exit 137) or even answer from the WRONG rootfs — the
    Alpine agent env before the image overlay is pivoted, where a clean
    `command -v python3` rc=1 is a lie. No early exit on "absent": only a
    full deadline of absence condemns the image."""
    deadline = time.monotonic() + deadline_s
    while True:
        proc = backend.execute(name, ["sh", "-c", "command -v python3"], check=False)
        if proc.returncode == 0:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)


def build_stage_tar() -> str:
    """One tarball with everything the guest runtime needs:

      runner.py        — the worker server (GUEST_RUNNER)
      cloudpickle/     — the HOST's cloudpickle, so by-value pickle streams
                         always match the version that produced them (no pip,
                         no network needed in the guest)
      fleetlet/       — this package, so synced project code can
                         `import fleetlet` inside the VM (decorators become
                         inert handles there, like modal-in-the-container)
    """
    import os
    import tarfile
    import tempfile

    import fleetlet

    fd, path = tempfile.mkstemp(prefix="fleetlet-stage-", suffix=".tgz")
    os.close(fd)

    def add_package(tar: tarfile.TarFile, pkg_dir: str, arcname: str) -> None:
        for entry in sorted(os.listdir(pkg_dir)):
            if entry.endswith(".py"):
                tar.add(os.path.join(pkg_dir, entry), arcname=f"{arcname}/{entry}")

    with tarfile.open(path, "w:gz") as tar:
        pkg_dir = os.path.dirname(fleetlet.__file__)
        tar.add(os.path.join(pkg_dir, "_runner.py"), arcname="runner.py")
        add_package(tar, pkg_dir, "fleetlet")
        add_package(tar, os.path.dirname(cloudpickle.__file__), "cloudpickle")
    return path
