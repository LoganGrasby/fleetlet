"""Fast unit tests — no smolvm, no VMs. Run: pytest tests/test_unit.py"""

import io
import json
import os
import pickle
import socket
import threading
import urllib.error
import urllib.request
from typing import Literal, Optional

import cloudpickle
import pytest

import fleetlet
from fleetlet import _proto
from fleetlet._function import Options, _map_pool_size, main_file_rel, resolve_spec, slugify
from fleetlet._machine import MachineSpec
from fleetlet._pool import PoolConfig, Task
from fleetlet._schema import fn_schema, validate
from fleetlet.errors import ConfigError
from fleetlet.image import Image


# ---------------------------------------------------------------- protocol

def test_frame_roundtrip():
    a, b = socket.socketpair()
    payload = {"op": "call", "args": b"\x00" * 100_000, "n": 42}
    thread = threading.Thread(target=_proto.send_frame, args=(a, payload))
    thread.start()
    received = _proto.recv_frame(b)
    thread.join()
    assert received == payload
    a.close()
    b.close()


def test_frame_eof_returns_none():
    a, b = socket.socketpair()
    a.close()
    assert _proto.recv_frame(b) is None
    b.close()


def test_runner_framing_matches_host():
    """The runner duplicates the framing code — keep the two in lockstep."""
    from fleetlet import _runner

    a, b = socket.socketpair()
    _proto.send_frame(a, {"op": "hello"})
    assert _runner.recv_frame(b) == {"op": "hello"}
    _runner.send_frame(b, {"ok": True})
    assert _proto.recv_frame(a) == {"ok": True}
    assert _runner.PROTO_VERSION == _proto.PROTO_VERSION
    a.close()
    b.close()


# ---------------------------------------------------------------- spec resolution

def module_level_fn(x):
    return x + 1


def test_resolve_importable_function():
    spec, blob = resolve_spec(module_level_fn, None)
    assert spec == {
        "kind": "import",
        "module": "test_unit",
        "qualname": "module_level_fn",
    }
    assert blob is None


def test_resolve_lambda_ships_blob():
    spec, blob = resolve_spec(lambda x: x * 2, None)
    assert spec["kind"] == "blob"
    assert blob is not None
    fn = pickle.loads(blob)
    assert fn(21) == 42


def test_resolve_closure_ships_blob():
    y = 10

    def closure(x):
        return x + y

    spec, blob = resolve_spec(closure, None)
    assert spec["kind"] == "blob"
    assert pickle.loads(blob)(5) == 15


# ---------------------------------------------------------------- runner logic

def test_runner_resolve_unwraps_decorated():
    from fleetlet._runner import Runner

    app = fleetlet.App("t", sync_project=False)
    decorated = app.function(module_level_fn)
    runner = Runner(project_root=None)
    resolved = runner.resolve(
        {"kind": "import", "module": "test_unit", "qualname": "module_level_fn"}
    )
    # The guest resolves the *decorated* symbol, then unwraps to the raw fn.
    assert resolved is module_level_fn
    assert decorated._fleetlet_raw is module_level_fn


def test_runner_enter_hooks_ordering():
    from fleetlet._runner import Runner

    calls = []

    class Base:
        @fleetlet.enter
        def warm_base(self):
            calls.append("base")

    class Actor(Base):
        @fleetlet.enter
        def warm(self):
            calls.append("actor")

        def work(self):
            return "ok"

    runner = Runner(project_root=None)
    runner.instance = Actor()
    runner.run_enter_hooks()
    assert calls == ["base", "actor"]


def test_runner_call_captures_output_and_errors():
    from fleetlet._runner import Runner

    runner = Runner(project_root=None)
    runner.log = io.StringIO()

    def noisy(x):
        print("working on", x)
        if x < 0:
            raise ValueError("negative!")
        return x * 2

    blob = cloudpickle.dumps(noisy)
    spec = {"kind": "blob", "blob_id": "abc123"}
    ok = runner.op_call(
        {"spec": spec, "blob": blob, "args": pickle.dumps(((21,), {}))}
    )
    assert ok["ok"] and pickle.loads(ok["value"]) == 42
    assert "working on 21" in ok["stdout"]

    err = runner.op_call(
        {"spec": spec, "args": pickle.dumps(((-1,), {}))}  # blob cached now
    )
    assert not err["ok"]
    assert err["error"]["etype"] == "ValueError"
    assert isinstance(pickle.loads(err["error"]["pickled"]), ValueError)


# ---------------------------------------------------------------- image

def test_image_immutable_chaining():
    base = Image.from_registry("python:3.12-slim")
    derived = base.pip_install("numpy").env(TZ="UTC").workdir("/app")
    assert base.steps == ()
    assert len(derived.steps) == 1 and "numpy" in derived.steps[0]
    assert derived.env_vars == (("TZ", "UTC"),)
    assert derived.content_id != base.content_id
    assert derived.needs_network and not base.needs_network


def test_image_from_smolmachine_creates_with_from_flag():
    img = Image.from_smolmachine("dist/py314.smolmachine")
    assert img.base.endswith("/dist/py314.smolmachine")  # absolutized
    args = MachineSpec(name="m", image=img.base).create_args()
    assert "--from" in args and "--image" not in args
    # Registry tags keep using --image.
    args = MachineSpec(name="m", image="python:3.14-slim").create_args()
    assert "--image" in args and "--from" not in args


def test_default_image_env_override(monkeypatch):
    monkeypatch.setenv("FLEETLET_DEFAULT_IMAGE", "python:3.12-alpine")
    assert Image.default().base == "python:3.12-alpine"
    monkeypatch.setenv("FLEETLET_DEFAULT_IMAGE", "cache/py.smolmachine")
    assert Image.default().base.endswith("/cache/py.smolmachine")
    monkeypatch.delenv("FLEETLET_DEFAULT_IMAGE")
    assert Image.default().base.startswith("python:3.")


def test_net_false_with_build_steps_rejected():
    cfg = PoolConfig(
        app_name="a", fn_slug="f", run_id="r",
        image=Image.default().pip_install("numpy"), net=False,
    )
    with pytest.raises(ConfigError):
        cfg.validate()


# ---------------------------------------------------------------- misc

def test_machine_spec_args():
    spec = MachineSpec(
        name="m", image="alpine", cpus=2, memory_mib=512,
        ports=[(42001, 7777)], net=True, cuda=True,
    )
    args = spec.create_args()
    assert "--net" in args and "--cuda" in args
    assert args[args.index("-p") + 1] == "42001:7777"


def test_slugify():
    assert slugify("My Función!") == "my-funci-n"
    assert slugify("__init__") == "init"


def test_map_pool_size():
    assert _map_pool_size(Options(workers=3), 30) == 3      # explicit wins
    assert _map_pool_size(Options(workers=1), 30) == 1      # explicit 1 wins too
    assert _map_pool_size(Options(), 2) == 2                # autoscale to items
    assert _map_pool_size(Options(), 100) == 4              # capped default


def test_machine_prefix_collision_proof():
    kw = dict(app_name="app", run_id="r1", image=Image.default())
    a = PoolConfig(fn_slug="resize-images-small", **kw).machine_prefix()
    b = PoolConfig(fn_slug="resize-images-large", **kw).machine_prefix()
    assert a != b                       # 12-char slug truncation must not collide
    assert not a.startswith(b) and not b.startswith(a)  # teardown sweeps by prefix
    assert a == PoolConfig(fn_slug="resize-images-small", **kw).machine_prefix()
    assert a.startswith("flt-")


def test_pool_mode_resolution():
    kw = dict(app_name="a", fn_slug="f", run_id="r", image=Image.default())
    assert PoolConfig(**kw, size=1).resolved_mode() == "cold"
    assert PoolConfig(**kw, size=4).resolved_mode() == "fork"
    assert PoolConfig(**kw, size=1, mode="fork").resolved_mode() == "fork"


def test_call_on_function_gives_hint():
    app = fleetlet.App("t", sync_project=False)
    fn = app.function(module_level_fn)
    with pytest.raises(ConfigError, match="remote"):
        fn(1)


def test_remote_outside_run_raises():
    app = fleetlet.App("t", sync_project=False)
    fn = app.function(module_level_fn)
    with pytest.raises(fleetlet.AppNotRunning):
        fn.remote(1)


# ---------------------------------------------------------------- http schema

def test_fn_schema_from_hints():
    def fn(text: str, dims: int = 8, scale: float = 1.0, tags: list[str] = None): ...
    schema = fn_schema(fn)
    assert schema["properties"]["text"] == {"type": "string"}
    assert schema["properties"]["dims"] == {"type": "integer", "default": 8}
    assert schema["properties"]["tags"]["items"] == {"type": "string"}
    assert schema["required"] == ["text"]
    assert schema["additionalProperties"] is False


def test_fn_schema_optional_literal_and_unknown():
    class Exotic: ...
    def fn(mode: Literal["a", "b"], note: Optional[str] = None, blob: Exotic = None): ...
    schema = fn_schema(fn)
    assert schema["properties"]["mode"] == {"enum": ["a", "b"]}
    assert schema["properties"]["note"]["anyOf"] == [{"type": "string"}, {"type": "null"}]
    assert schema["properties"]["blob"] == {"default": None}  # unknown → anything


def test_fn_schema_var_kwargs_opens_extras():
    def fn(a: int, **rest): ...
    schema = fn_schema(fn)
    assert schema["additionalProperties"] is True
    kwargs, errors = validate(schema, {"a": 1, "anything": "goes"})
    assert not errors and kwargs["anything"] == "goes"


def test_validate_coercion_and_errors():
    def fn(n: int, x: float, name: str = "hi"): ...
    schema = fn_schema(fn)

    kwargs, errors = validate(schema, {"n": 3, "x": 2})
    assert not errors and kwargs == {"n": 3, "x": 2.0} and isinstance(kwargs["x"], float)

    _, errors = validate(schema, {"n": True, "x": 1.5})
    assert any("integer" in e for e in errors)          # bool is not an int
    _, errors = validate(schema, {"x": 1.5})
    assert any("missing required" in e for e in errors)
    _, errors = validate(schema, {"n": 1, "x": 1.5, "zzz": 1})
    assert any("unknown parameter" in e for e in errors)
    _, errors = validate(schema, [1, 2])
    assert any("JSON object" in e for e in errors)


# ---------------------------------------------------------------- http gateway

@pytest.fixture()
def http_service():
    from fleetlet.serve import make_server

    app = fleetlet.App("t", sync_project=False)

    @app.function()
    def add(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @app.function()
    def boom(msg: str) -> None:
        raise ValueError(msg)

    server = make_server(app, "127.0.0.1", 0, local=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        server.gateway.close()


def _http(method, url, body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, json.loads(payload) if payload else {}


def test_gateway_index_call_and_validation(http_service):
    status, index = _http("GET", http_service + "/")
    assert status == 200
    assert index["functions"]["add"]["doc"] == "Add two numbers."
    assert index["functions"]["add"]["params"]["required"] == ["a"]

    status, body = _http("POST", http_service + "/call/add", {"a": 2, "b": 40})
    assert (status, body["result"]) == (200, 42)
    status, body = _http("POST", http_service + "/call/add", {"a": 2})
    assert (status, body["result"]) == (200, 3)  # default b=1

    status, body = _http("POST", http_service + "/call/add", {"a": "two"})
    assert status == 400 and "integer" in body["error"]["message"]
    status, _ = _http("POST", http_service + "/call/nope", {})
    assert status == 404

    # bare /<fn> is everyone's first guess — the 404 must point at /call/<fn>
    status, body = _http("POST", http_service + "/add", {"a": 1})
    assert status == 404 and "did you mean POST /call/add" in body["error"]["message"]


def test_gateway_spawn_result_and_errors(http_service):
    status, body = _http("POST", http_service + "/spawn/add", {"a": 20, "b": 22})
    assert status == 202
    url = http_service + body["result"]
    for _ in range(50):
        status, body = _http("GET", url)
        if status == 200:
            break
    assert body["result"] == 42
    status, _ = _http("GET", url)
    assert status == 404  # results are handed out once

    status, body = _http("POST", http_service + "/call/boom", {"msg": "nope"})
    assert status == 500 and body["error"]["type"] == "ValueError"


# ---------------------------------------------------------------- cloud backend

class FakeCloudClient:
    """Records API calls; answers like the live control plane."""

    api_key = "smk_test"

    def __init__(self):
        self.machines: dict[str, dict] = {}
        self.calls: list[tuple] = []
        self.seq = 0

    def create_machine(self, **kw):
        self.seq += 1
        mid = f"mach-{self.seq:04d}"
        self.calls.append(("create", kw))
        info = {"id": mid, "name": kw["name"], "state": "stopped",
                "url": f"https://{kw['name']}-t.apps.example"}
        self.machines[mid] = info
        return info

    def get_machine(self, mid):
        return self.machines[mid]

    def list_machines(self):
        return list(self.machines.values())

    def start_machine(self, mid):
        self.machines[mid]["state"] = "started"

    def stop_machine(self, mid):
        self.machines[mid]["state"] = "stopped"

    def delete_machine(self, mid):
        del self.machines[mid]

    def wait_ready(self, mid, deadline_s=0):
        return self.machines[mid]

    def wait_url(self, mid, deadline_s=0):
        return self.machines[mid]["url"]

    def exec(self, mid, command, **kw):
        self.calls.append(("exec", mid, command, kw))
        return {"exitCode": 0, "stdout": "ok", "stderr": ""}

    def fork_machine(self, golden_id, name, guest_ports=None):
        self.calls.append(("fork", golden_id, name, guest_ports))
        self.seq += 1
        mid = f"mach-{self.seq:04d}"
        self.machines[mid] = {"id": mid, "name": name, "state": "started", "url": None}
        return self.machines[mid]

    def upload_file(self, mid, path, data):
        self.calls.append(("upload", mid, path, len(data)))


def _cloud_backend():
    from fleetlet._backend import CloudBackend

    return CloudBackend(FakeCloudClient())


def test_cloud_backend_create_maps_spec():
    backend = _cloud_backend()
    backend.create(MachineSpec(name="m1", image="python:3.12-slim", cpus=2,
                               memory_mib=768, ports=[(0, 7777)], net=True))
    _, kw = backend.client.calls[0]
    assert kw["guest_ports"] == [7777]
    assert kw["memory_mb"] == 768 and kw["cpus"] == 2
    assert kw["network"] == {"mode": "open"}
    assert kw["ttl_seconds"]  # leak safety net on by default
    assert backend.status("m1")["state"] == "stopped"
    backend.start("m1")
    # cloud "started" is normalized to the local vocabulary
    assert backend.status("m1")["state"] == "running"


def test_cloud_backend_rejects_local_only_options():
    backend = _cloud_backend()
    base = dict(image="python:3.12-slim", cpus=1, memory_mib=256)
    with pytest.raises(ConfigError, match="volumes"):
        backend.create(MachineSpec(name="m", volumes=["/a:/b"], **base))
    with pytest.raises(ConfigError, match="local-only"):
        backend.create(MachineSpec(name="m", cuda=True, **base))
    with pytest.raises(ConfigError, match="registry"):
        backend.create(MachineSpec(name="m", image="dist/py.smolmachine",
                                   cpus=1, memory_mib=256))


def test_cloud_backend_detach_wraps_nohup():
    backend = _cloud_backend()
    backend.create(MachineSpec(name="m1", image="python:3.12-slim", cpus=1, memory_mib=256))
    backend.execute("m1", ["python3", "runner.py", "--http"], detach=True)
    _, _, command, _ = backend.client.calls[-1]
    assert command[0] == "sh" and "nohup python3 runner.py --http" in command[2]


def test_cloud_backend_fork_flow():
    backend = _cloud_backend()
    spec = MachineSpec(name="g", image="python:3.12-slim", cpus=1, memory_mib=256,
                       ports=[(0, 7777)])
    spec.forkable = True
    backend.create(spec)
    _, kw = backend.client.calls[0]
    assert kw["forkable"] is True
    backend.start("g", forkable=True)  # no-op flag on cloud, must not raise
    backend.fork("g", "w0", ports=[(0, 7777)], share_weights=False)
    call = next(c for c in backend.client.calls if c[0] == "fork")
    assert call[2] == "w0" and call[3] == [7777]
    assert backend.status("w0")["state"] == "running"


def test_exec_relay_round_trip_against_real_runner():
    """RELAY_SRC (the in-guest bridge) must speak the socket runner's real
    framing: run the actual runner locally, pipe a hello op through the relay
    exactly as the cloud exec would (stdin b64 in, 'B64:' stdout out)."""
    import subprocess
    import time as _time

    from fleetlet._pool import RELAY_SRC
    from fleetlet._runner import Runner

    runner = Runner(project_root=None)
    runner.log = open(os.devnull, "w")
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    threading.Thread(target=runner.serve, args=("127.0.0.1", port), daemon=True).start()
    _time.sleep(0.3)

    import base64
    payload = base64.b64encode(pickle.dumps({"op": "hello"}))
    proc = subprocess.run(
        [__import__("sys").executable, "-c", RELAY_SRC.format(port=port)],
        input=payload, capture_output=True, timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.decode()
    assert out.startswith("B64:")
    resp = pickle.loads(base64.b64decode(out[4:]))
    assert resp["ok"] and resp["proto"] == 1


def test_cloud_backend_deployment_ttl_opt_out():
    backend = _cloud_backend()
    backend.create(MachineSpec(name="d1", image="python:3.12-slim", cpus=1,
                               memory_mib=256, ttl_seconds=0))
    _, kw = backend.client.calls[0]
    assert kw["ttl_seconds"] is None


def test_resolve_target(monkeypatch):
    from fleetlet._backend import resolve_target

    monkeypatch.delenv("FLEETLET_TARGET", raising=False)
    assert resolve_target(None) == "local"
    monkeypatch.setenv("FLEETLET_TARGET", "cloud")
    assert resolve_target(None) == "cloud"
    assert resolve_target("local") == "local"  # explicit beats env
    with pytest.raises(ConfigError):
        resolve_target("k8s")


# ---------------------------------------------------------------- runner http

@pytest.fixture()
def http_runner():
    from fleetlet._runner import Runner, serve_http

    runner = Runner(project_root=None)
    runner.log = open(os.devnull, "w")
    # Pick a free port, then hand it to the server thread.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    thread = threading.Thread(target=serve_http,
                              args=(runner, "127.0.0.1", port, "tok"), daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/healthz", timeout=1)
            break
        except OSError:
            import time
            time.sleep(0.05)
    yield base


def _op(base, payload, token="tok"):
    req = urllib.request.Request(base + "/submit", data=pickle.dumps(payload),
                                 headers={"x-fleetlet-token": token}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        job = json.loads(resp.read())["id"]
    req = urllib.request.Request(base + f"/poll?id={job}&wait=10",
                                 headers={"x-fleetlet-token": token})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return pickle.loads(resp.read())


def test_runner_http_submit_poll_and_auth(http_runner):
    resp = _op(http_runner, {"op": "hello"})
    assert resp["ok"] and resp["proto"] == 1

    # wrong token -> 403
    req = urllib.request.Request(http_runner + "/submit", data=b"x",
                                 headers={"x-fleetlet-token": "nope"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(req, timeout=5)
    assert err.value.code == 403

    # a real call through the HTTP framing
    blob = cloudpickle.dumps(lambda x: x * 3)
    resp = _op(http_runner, {"op": "call", "spec": {"kind": "blob", "blob_id": "b1"},
                             "blob": blob, "args": pickle.dumps(((14,), {}))})
    assert resp["ok"] and pickle.loads(resp["value"]) == 42


def test_http_worker_rpc_against_real_runner(http_runner):
    from fleetlet._pool import HttpWorker

    class StubBackend:
        api_key = "smk_test"

        def status(self, name):
            return {"state": "running"}

        def ingress_url(self, name, deadline_s=None):
            return http_runner

    worker = HttpWorker("w0", StubBackend(), "tok")
    worker.wait_ready(10)
    assert worker.guest_py  # hello() populated it

    blob = cloudpickle.dumps(lambda a, b: a + b)
    result = worker.call(Task(
        spec={"kind": "blob", "blob_id": "b2"}, blob=blob,
        args_blob=cloudpickle.dumps(((20, 22), {})), timeout=15, retries=0, tag="t",
    ))
    assert result == 42


# ---------------------------------------------------------------- deploy guards

def test_deploy_guards(monkeypatch, tmp_path):
    from fleetlet import _deploy

    class GuardStop(Exception):
        pass

    class NoBackend:
        def list_machines(self):
            raise GuardStop  # guards must fire before any API traffic

    monkeypatch.setattr(_deploy, "get_backend", lambda t: NoBackend())

    app = fleetlet.App("t", sync_project=False)
    with pytest.raises(ConfigError, match="no @app.function"):
        _deploy.deploy(app, "x.py")

    app2 = fleetlet.App("t2", sync_project=False)
    app2.function(module_level_fn)
    with pytest.raises(ConfigError, match="sync_project"):
        _deploy.deploy(app2, "x.py")

    app3 = fleetlet.App("t3", project_root=str(tmp_path))
    app3.function(module_level_fn)
    with pytest.raises(ConfigError, match="outside the project root"):
        _deploy.deploy(app3, "/etc/hosts")


def test_deploy_aborts_when_existing_delete_fails(monkeypatch, tmp_path):
    from fleetlet import _deploy
    from fleetlet.errors import CloudError

    class StuckBackend:
        def list_machines(self):
            return [{"name": "flt-deploy-svc", "state": "running"}]

        def stop(self, name):
            return True

        def delete(self, name):
            return False  # e.g. the API 500s — the old machine still exists

    monkeypatch.setattr(_deploy, "get_backend", lambda t: StuckBackend())
    (tmp_path / "svc.py").write_text("")
    app = fleetlet.App("svc", project_root=str(tmp_path))
    app.function(module_level_fn)
    with pytest.raises(CloudError, match="could not delete"):
        _deploy.deploy(app, str(tmp_path / "svc.py"))


# ---------------------------------------------------------------- error paths

def test_cloud_backend_list_error_propagates_but_teardown_stays_besteffort():
    from fleetlet.errors import CloudError

    backend = _cloud_backend()

    def boom():
        raise CloudError(401, "bad key")

    backend.client.list_machines = boom
    with pytest.raises(CloudError):        # an API failure is not an empty tenant
        backend.list_machines()
    assert backend.stop("ghost") is False  # cleanup helpers must not raise
    assert backend.delete("ghost") is False
    assert backend.status("ghost") is None


def test_wait_ready_permanent_errors(monkeypatch):
    from fleetlet import _cloud
    from fleetlet.errors import CloudError

    client = _cloud.CloudClient(api_key="smk_x", base_url="https://api.example")

    def raise_status(status):
        def _get(mid):
            raise CloudError(status, "nope")
        return _get

    monkeypatch.setattr(client, "get_machine", raise_status(401))
    with pytest.raises(CloudError, match="401"):   # immediate, no 180s spin
        client.wait_ready("m1", deadline_s=30)

    monkeypatch.setattr(_cloud, "NOT_FOUND_GRACE", 0.0)
    monkeypatch.setattr(client, "get_machine", raise_status(404))
    with pytest.raises(CloudError, match="404"):   # permanent once past grace
        client.wait_ready("m1", deadline_s=30)


def test_exec_relay_untimed_ops_surface_the_cap():
    from fleetlet._pool import ExecRelayWorker

    class Client:
        def exec(self, mid, cmd, stdin=None, timeout=None):
            return {"exitCode": 124, "stdout": "", "stderr": "command timed out"}

    class Backend:
        client = Client()

        def _require_id(self, name):
            return "id1"

    worker = ExecRelayWorker("w0", Backend())
    with pytest.raises(TimeoutError, match="bounds untimed ops"):
        worker.rpc({"op": "hello"}, None)
    with pytest.raises(TimeoutError, match="exceeded 5s"):
        worker.rpc({"op": "hello"}, 5.0)


def test_fn_schema_var_positional_rejects_unknown_params():
    def fn(x: int, *args): ...
    schema = fn_schema(fn)
    assert schema["additionalProperties"] is False  # *args can't take named extras
    _, errors = validate(schema, {"x": 1, "junk": 2})
    assert any("unknown parameter" in e for e in errors)  # 400, not a 500 TypeError


def test_gateway_result_ttl_eviction(monkeypatch):
    import concurrent.futures as cf

    from fleetlet import serve as serve_mod

    app = fleetlet.App("t", sync_project=False)
    app.function(module_level_fn)
    gateway = serve_mod.Gateway(app, local=True)
    try:
        future: cf.Future = cf.Future()
        call_id = gateway.store_call(future)
        future.set_result(41)                # completion timestamps the entry
        assert call_id in gateway.calls
        monkeypatch.setattr(serve_mod, "RESULT_TTL", 0.0)
        gateway.store_call(cf.Future())      # any store/lookup sweeps expired entries
        assert call_id not in gateway.calls
        running = cf.Future()
        running_id = gateway.store_call(running)
        gateway.store_call(cf.Future())
        assert running_id in gateway.calls   # running calls are never evicted
    finally:
        gateway.close()


def test_cli_fleet_machines_app_matches_deployment():
    from fleetlet import cli

    class FakeBackend:
        def list_machines(self):
            return [
                {"name": "flt-embeds-embed-abc123-dead01-w0", "state": "running"},
                {"name": "flt-deploy-embeds", "state": "running"},
                {"name": "flt-deploy-embedsvc", "state": "running"},
                {"name": "flt-other-fn-abc123-dead01-w0", "state": "running"},
            ]

    names = {m["name"] for m in cli._fleet_machines(FakeBackend(), "embeds")}
    # The command cmd_deploy prints (`clean --deployments --app <slug>`) must
    # reach the deployment itself — by exact name, not loose prefix.
    assert names == {"flt-embeds-embed-abc123-dead01-w0", "flt-deploy-embeds"}
    assert len(cli._fleet_machines(FakeBackend())) == 4


def _bare_pool(**backend_attrs):
    import types

    from fleetlet._pool import Pool

    cfg = PoolConfig(app_name="a", fn_slug="f", run_id="r", image=Image.default())
    backend = types.SimpleNamespace(target="local", list_machines=lambda: [],
                                    **backend_attrs)
    return Pool(cfg, backend=backend)


def test_pool_drain_fails_queued_tasks_but_keeps_sentinels():
    import queue as queue_mod

    from fleetlet._pool import _SENTINEL, Task
    from fleetlet.errors import WorkerError

    pool = _bare_pool()
    task = Task(spec={}, blob=None, args_blob=b"", timeout=None, retries=0, tag="t")
    pool.tasks.put(task)
    pool.tasks.put(_SENTINEL)
    pool._fail_queued_tasks("last worker gone")
    with pytest.raises(WorkerError, match="last worker gone"):
        task.future.result(timeout=1)
    assert pool.tasks.get_nowait() is _SENTINEL  # sentinels survive the drain
    with pytest.raises(queue_mod.Empty):
        pool.tasks.get_nowait()


def test_pool_shutdown_fails_pending_tasks():
    from fleetlet._pool import Task
    from fleetlet.errors import WorkerError

    pool = _bare_pool()
    task = Task(spec={}, blob=None, args_blob=b"", timeout=None, retries=1, tag="t")
    pool.tasks.put(task)
    pool.shutdown()
    # A queued-but-never-served task must fail fast, not hang its waiter.
    with pytest.raises(WorkerError, match="shut down"):
        task.future.result(timeout=1)
