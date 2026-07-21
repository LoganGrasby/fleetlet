"""Where machines live: the local smolvm engine, or the smol cloud.

One `Backend` interface over both, so pools and the CLI never care which
side they're driving:

* ``LocalBackend`` — delegates to `_machine` (the smolvm CLI wrapper).
  Workers are reached over forwarded TCP ports; fork pools supported.

* ``CloudBackend`` — drives the smol cloud REST API via `_cloud`.
  Machines are id-addressed there but name-addressed here, so the backend
  keeps a name→id map (refreshed from listing for machines it didn't
  create). Workers are reached over their Bearer-authed ingress URL.

Cloud platform behavior this design depends on:
* Background processes survive `exec` returning — a plain
  ``nohup … &`` wrapper substitutes for local ``exec --detach``.
* A fork clone keeps the golden's running processes, but gets no ingress
  URL and new execs see a fresh container overlay — so fork-pool calls
  ride the exec API relay instead of HTTP.
* Machines default to 1 cpu / 256MB / network blocked; egress is opt-in.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from . import _machine
from ._cloud import CloudClient
from ._machine import MachineSpec
from ._proto import GUEST_LOG
from .errors import CloudError, ConfigError, SmolvmError

TARGET_ENV = "FLEETLET_TARGET"
# Safety net: if the host process dies without teardown, cloud workers would
# otherwise leak (and bill) forever. Deploys pass ttl_seconds=0 (= no TTL).
WORKER_TTL_ENV = "FLEETLET_CLOUD_TTL"
DEFAULT_WORKER_TTL = 3600


@dataclass
class ExecResult:
    """Normalized exec outcome (mirrors the CompletedProcess fields we use)."""

    returncode: int
    stdout: bytes
    stderr: bytes


class Backend(Protocol):
    target: str

    def create(self, spec: MachineSpec) -> None: ...
    def start(self, name: str, *, forkable: bool = False) -> None: ...
    def fork(self, golden: str, name: str, *, ports: list[tuple[int, int]] | None,
             share_weights: bool) -> None: ...
    def execute(self, name: str, command: list[str], *, env: dict[str, str] | None = None,
                workdir: str | None = None, detach: bool = False,
                timeout: float = _machine.EXEC_TIMEOUT, check: bool = True) -> ExecResult: ...
    def put_file(self, name: str, local_path: str, guest_path: str) -> None: ...
    def stop(self, name: str) -> bool: ...
    def delete(self, name: str) -> bool: ...
    def status(self, name: str) -> dict[str, Any] | None: ...
    def list_machines(self) -> list[dict[str, Any]]: ...


def resolve_target(target: str | None) -> str:
    resolved = (target or os.environ.get(TARGET_ENV) or "local").strip().lower()
    if resolved not in ("local", "cloud"):
        raise ConfigError(f"unknown target '{resolved}' — use 'local' or 'cloud'")
    return resolved


def get_backend(target: str | None = None) -> "Backend":
    if resolve_target(target) == "cloud":
        return CloudBackend(CloudClient())
    return LocalBackend()


# ---------------------------------------------------------------------------
# Local (smolvm CLI)
# ---------------------------------------------------------------------------
class LocalBackend:
    target = "local"

    def create(self, spec: MachineSpec) -> None:
        _machine.create(spec)

    def start(self, name: str, *, forkable: bool = False) -> None:
        _machine.start(name, forkable=forkable)

    def fork(self, golden: str, name: str, *, ports=None, share_weights=False) -> None:
        _machine.fork(golden, name, ports=ports, share_weights=share_weights)

    def execute(self, name, command, *, env=None, workdir=None, detach=False,
                timeout=_machine.EXEC_TIMEOUT, check=True) -> ExecResult:
        proc: subprocess.CompletedProcess = _machine.execute(
            name, command, env=env, workdir=workdir, detach=detach,
            timeout=timeout, check=check,
        )
        return ExecResult(proc.returncode, proc.stdout or b"", proc.stderr or b"")

    def put_file(self, name: str, local_path: str, guest_path: str) -> None:
        _machine.cp_to(name, local_path, guest_path)

    def stop(self, name: str) -> bool:
        return _machine.stop(name)

    def delete(self, name: str) -> bool:
        return _machine.delete(name)

    def status(self, name: str) -> dict[str, Any] | None:
        return _machine.status(name)

    def list_machines(self) -> list[dict[str, Any]]:
        return _machine.list_machines()


# ---------------------------------------------------------------------------
# Cloud (smol cloud /v1)
# ---------------------------------------------------------------------------
class CloudBackend:
    target = "cloud"

    def __init__(self, client: CloudClient):
        self.client = client
        self._ids: dict[str, str] = {}
        self._ids_lock = threading.Lock()

    # ------------------------------------------------------------ naming

    def _remember(self, name: str, machine_id: str) -> None:
        with self._ids_lock:
            self._ids[name] = machine_id

    def _id_for(self, name: str, *, refresh: bool = True) -> str | None:
        with self._ids_lock:
            if name in self._ids:
                return self._ids[name]
        if not refresh:
            return None
        for m in self.list_machines():  # refreshes the map as a side effect
            if m.get("name") == name:
                return m["id"]
        return None

    def _require_id(self, name: str) -> str:
        machine_id = self._id_for(name)
        if machine_id is None:
            raise SmolvmError(["cloud", name], 1, f"unknown cloud machine '{name}'")
        return machine_id

    # ------------------------------------------------------------ lifecycle

    def create(self, spec: MachineSpec) -> None:
        if spec.volumes:
            raise ConfigError(
                "volumes= mounts host directories, which don't exist on the cloud "
                "target — drop volumes or run locally"
            )
        if spec.gpu or spec.cuda:
            raise ConfigError("gpu/cuda are local-only options (no cloud GPU nodes yet)")
        if spec.image.endswith(".smolmachine"):
            raise ConfigError(
                "packed .smolmachine artifacts are local files — the cloud target "
                "needs a registry image reference (e.g. python:3.12-slim)"
            )
        network = None
        if spec.allow_hosts:
            # Host-scoped egress wire shape: the empty cidrs list is intentional.
            network = {"mode": "allowCidrs", "cidrs": [], "hosts": spec.allow_hosts}
        elif spec.net:
            network = {"mode": "open"}
        created = self.client.create_machine(
            name=spec.name,
            image=spec.image,
            cpus=spec.cpus,
            memory_mb=spec.memory_mib,
            guest_ports=[guest for _, guest in spec.ports],
            env=spec.env,
            network=network,
            disk_gb=spec.storage_gb,
            forkable=spec.forkable,
            ttl_seconds=(worker_ttl() if spec.ttl_seconds is None
                         else (spec.ttl_seconds or None)),
        )
        self._remember(spec.name, created["id"])

    def start(self, name: str, *, forkable: bool = False) -> None:
        # Cloud forkability is a create-body property (MachineSpec.forkable);
        # by start time it's already decided, so the flag is a no-op here.
        machine_id = self._require_id(name)
        self.client.start_machine(machine_id)
        self.client.wait_ready(machine_id)

    def fork(self, golden, name, *, ports=None, share_weights=False) -> None:
        """Live-RAM CoW clone on the golden's node. The clone keeps the
        golden's running processes — but new execs see a fresh container
        overlay, and no ingress URL is provisioned for clones, which is
        why clone traffic rides the exec API instead."""
        golden_id = self._require_id(golden)
        clone = self.client.fork_machine(
            golden_id, name, guest_ports=[g for _, g in (ports or [])]
        )
        self._remember(name, clone["id"])
        self.client.wait_ready(clone["id"])

    def execute(self, name, command, *, env=None, workdir=None, detach=False,
                timeout=_machine.EXEC_TIMEOUT, check=True) -> ExecResult:
        machine_id = self._require_id(name)
        if detach:
            # No --detach equivalent; a backgrounded child survives the exec
            # returning, so wrap in nohup + & ourselves.
            command = ["sh", "-c",
                       f"nohup {shlex.join(command)} >>{GUEST_LOG} 2>&1 & echo detached"]
        try:
            resp = self.client.exec(
                machine_id, command, env=env, workdir=workdir, timeout=timeout,
            )
        except CloudError as exc:
            if check:
                raise SmolvmError(["cloud", "exec", name, *command], 1, str(exc)) from None
            return ExecResult(1, b"", str(exc).encode())
        result = ExecResult(
            returncode=int(resp.get("exitCode", 0)),
            stdout=str(resp.get("stdout", "")).encode(),
            stderr=str(resp.get("stderr", "")).encode(),
        )
        if check and result.returncode != 0:
            raise SmolvmError(
                ["cloud", "exec", name, *command], result.returncode,
                result.stderr.decode(errors="replace"),
            )
        return result

    def put_file(self, name: str, local_path: str, guest_path: str) -> None:
        machine_id = self._require_id(name)
        with open(local_path, "rb") as fh:
            self.client.upload_file(machine_id, guest_path, fh.read())

    def stop(self, name: str) -> bool:
        # stop/delete/status are best-effort teardown helpers: an API failure
        # (including one inside the _id_for listing refresh) means False/None,
        # never an exception out of a cleanup path.
        try:
            machine_id = self._id_for(name)
        except CloudError:
            return False
        if machine_id is None:
            return False
        try:
            self.client.stop_machine(machine_id)
            return True
        except CloudError:
            return False

    def delete(self, name: str) -> bool:
        try:
            machine_id = self._id_for(name)
        except CloudError:
            return False
        if machine_id is None:
            return False
        try:
            self.client.delete_machine(machine_id)
        except CloudError as exc:
            if exc.status != 404:
                return False
        with self._ids_lock:
            self._ids.pop(name, None)
        return True

    def status(self, name: str) -> dict[str, Any] | None:
        try:
            machine_id = self._id_for(name)
        except CloudError:
            return None
        if machine_id is None:
            return None
        try:
            info = self.client.get_machine(machine_id)
        except CloudError:
            return None
        return self._normalize(info)

    def list_machines(self) -> list[dict[str, Any]]:
        # Deliberately propagates CloudError: an API/auth failure is not an
        # empty tenant, and `ls`/`clean` reporting "nothing to clean" on a bad
        # token would mask machines that keep running (and billing).
        machines = self.client.list_machines()
        rows = []
        with self._ids_lock:
            for m in machines:
                if m.get("name"):
                    self._ids[m["name"]] = m["id"]
                rows.append(self._normalize(m))
        return [r for r in rows if r.get("name")]

    # ------------------------------------------------------------ cloud extras

    def ingress_url(self, name: str, deadline_s: float | None = None) -> str:
        machine_id = self._require_id(name)
        if deadline_s is None:
            return self.client.wait_url(machine_id)
        return self.client.wait_url(machine_id, deadline_s)

    @property
    def api_key(self) -> str:
        return self.client.api_key

    @staticmethod
    def _normalize(info: dict[str, Any]) -> dict[str, Any]:
        """Cloud says "started" where the local CLI says "running" — liveness
        checks compare against "running", so map it."""
        out = dict(info)
        if out.get("state") in ("started",):
            out["state"] = "running"
        return out


def worker_ttl() -> int | None:
    """TTL applied to cloud worker machines (server deletes them after this
    many seconds even if we crash). 0 disables."""
    raw = os.environ.get(WORKER_TTL_ENV, "").strip()
    if not raw:
        return DEFAULT_WORKER_TTL
    ttl = int(raw)
    return ttl if ttl > 0 else None
