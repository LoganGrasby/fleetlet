"""Thin typed wrapper over the `smolvm machine` CLI.

Every fleetlet worker is a smolvm machine; this module is the only place
that shells out. All functions raise SmolvmError on non-zero exit.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

from .errors import SmolvmError, FleetletError

SMOLVM = "smolvm"

# Generous defaults: create/start may pull images, exec may run pip installs.
CREATE_TIMEOUT = 600
START_TIMEOUT = 600
EXEC_TIMEOUT = 1800
FAST_TIMEOUT = 120


def smolvm_available() -> str | None:
    """Return the smolvm binary path, or None if not installed."""
    return shutil.which(SMOLVM)


def _run(
    args: list[str],
    *,
    timeout: float = FAST_TIMEOUT,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess:
    cmd = [SMOLVM, *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            input=input_bytes,
        )
    except FileNotFoundError:
        raise FleetletError(
            "smolvm binary not found. Install it first:\n"
            "  curl -sSL https://smolmachines.com/install.sh | bash"
        ) from None
    except subprocess.TimeoutExpired:
        raise SmolvmError(cmd, -1, f"timed out after {timeout}s") from None
    if check and proc.returncode != 0:
        raise SmolvmError(cmd, proc.returncode, proc.stderr.decode(errors="replace"))
    return proc


@dataclass
class MachineSpec:
    """Everything needed to `machine create` a worker or golden."""

    name: str
    image: str
    cpus: int = 2
    memory_mib: int = 1024
    ports: list[tuple[int, int]] = field(default_factory=list)  # (host, guest)
    volumes: list[str] = field(default_factory=list)  # "HOST:GUEST[:ro]"
    env: dict[str, str] = field(default_factory=dict)
    net: bool = False
    allow_hosts: list[str] = field(default_factory=list)
    gpu: bool = False
    gpu_vram_mib: int | None = None
    cuda: bool = False
    storage_gb: int | None = None
    overlay_gb: int | None = None
    # Cloud-only: server-side delete-after-N-seconds safety net. None = the
    # backend's default worker TTL, 0 = no TTL (deployments). Local: ignored.
    ttl_seconds: int | None = None
    # Cloud-only: fork goldens must be BORN forkable (create-body flag there;
    # locally forkability is chosen at `machine start --forkable`).
    forkable: bool = False

    def create_args(self) -> list[str]:
        # A `.smolmachine` path is a packed artifact (`smolvm pack create`):
        # boots from pre-extracted layers, no registry pull.
        source = (
            ["--from", self.image] if self.image.endswith(".smolmachine")
            else ["--image", self.image]
        )
        args = [
            "machine", "create",
            "--name", self.name,
            *source,
            "--cpus", str(self.cpus),
            "--mem", str(self.memory_mib),
        ]
        for host, guest in self.ports:
            args += ["-p", f"{host}:{guest}"]
        for vol in self.volumes:
            args += ["-v", vol]
        for key, val in self.env.items():
            args += ["-e", f"{key}={val}"]
        if self.allow_hosts:
            for host in self.allow_hosts:
                args += ["--allow-host", host]
        elif self.net:
            args += ["--net"]
        if self.gpu:
            args += ["--gpu"]
            if self.gpu_vram_mib:
                args += ["--gpu-vram", str(self.gpu_vram_mib)]
        if self.cuda:
            args += ["--cuda"]
        if self.storage_gb:
            args += ["--storage", str(self.storage_gb)]
        if self.overlay_gb:
            args += ["--overlay", str(self.overlay_gb)]
        return args


def create(spec: MachineSpec) -> None:
    _run(spec.create_args(), timeout=CREATE_TIMEOUT)


def start(name: str, *, forkable: bool = False) -> None:
    args = ["machine", "start", "--name", name]
    if forkable:
        args.append("--forkable")
    _run(args, timeout=START_TIMEOUT)


def fork(
    golden: str,
    name: str,
    *,
    ports: list[tuple[int, int]] | None = None,
    share_weights: bool = False,
) -> None:
    args = ["machine", "fork", "--golden", golden, "--name", name]
    for host, guest in ports or []:
        args += ["-p", f"{host}:{guest}"]
    if share_weights:
        args.append("--share-weights")
    _run(args, timeout=FAST_TIMEOUT)


def execute(
    name: str,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    detach: bool = False,
    timeout: float = EXEC_TIMEOUT,
    check: bool = True,
) -> subprocess.CompletedProcess:
    args = ["machine", "exec", "--name", name]
    for key, val in (env or {}).items():
        args += ["-e", f"{key}={val}"]
    if workdir:
        args += ["-w", workdir]
    if detach:
        args += ["--detach"]
    args += ["--", *command]
    return _run(args, timeout=timeout, check=check)


def cp_to(name: str, local_path: str, guest_path: str) -> None:
    _run(["machine", "cp", local_path, f"{name}:{guest_path}"], timeout=EXEC_TIMEOUT)


def cp_from(name: str, guest_path: str, local_path: str) -> None:
    _run(["machine", "cp", f"{name}:{guest_path}", local_path], timeout=EXEC_TIMEOUT)


def stop(name: str, *, check: bool = False) -> bool:
    proc = _run(["machine", "stop", "--name", name], timeout=FAST_TIMEOUT, check=check)
    return proc.returncode == 0


def delete(name: str, *, check: bool = False) -> bool:
    proc = _run(["machine", "delete", "--name", name, "--force"], timeout=FAST_TIMEOUT, check=check)
    return proc.returncode == 0


def status(name: str) -> dict[str, Any] | None:
    proc = _run(["machine", "status", "--name", name, "--json"], check=False)
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout.decode())
    except json.JSONDecodeError:
        return None


def list_machines() -> list[dict[str, Any]]:
    """All machines as dicts with at least `name` and `state`.

    Uses `machine ls --json`: the human table truncates names to ~15
    chars, which would make delete-by-name impossible.
    """
    proc = _run(["machine", "ls", "--json"], check=False)
    if proc.returncode != 0:
        return []
    try:
        rows = json.loads(proc.stdout.decode())
    except json.JSONDecodeError:
        return []
    return [row for row in rows if isinstance(row, dict) and "name" in row]
