"""Minimal client for the smol cloud ``/v1`` REST API (the platform's own
name for its cloud is "smolfleet" — no relation to this library).

The same API the `smol` SDK's cloud transport speaks (https://github.com/
smol-machines/smol), reimplemented here on the stdlib so fleetlet needs no
extra dependency and can use the parts of the current server contract the
published SDK doesn't cover yet: ``forkable`` in the create body, machine
listing (cleanup!), exec ``stdin``, and the async-provisioned ingress ``url``.

Auth is a tenant API key (``smk_…``) from ``SMOL_CLOUD_TOKEN`` — the same
variable the smol SDK and CLI use. Endpoint override: ``SMOL_CLOUD_URL``.

One non-obvious platform fact this module leans on: a
machine's ingress URL (``https://<name>-<tenant>.apps.smolmachines.com``,
routed to its first published port) requires the SAME Bearer key — worker
traffic over it is tenant-private by default.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from .errors import CloudError, ConfigError

DEFAULT_BASE_URL = "https://api.smolmachines.com"
TOKEN_ENV = "SMOL_CLOUD_TOKEN"
URL_ENV = "SMOL_CLOUD_URL"

API_TIMEOUT = 30.0
# Slack on top of a command's own timeout when sizing the exec HTTP read
# timeout.
EXEC_HTTP_HEADROOM = 30.0
READY_DEADLINE = 180.0
URL_DEADLINE = 120.0
# How long wait_ready tolerates a 404 before treating it as permanent — the
# control plane's GET can briefly lag a create/fork.
NOT_FOUND_GRACE = 10.0

# Machine states the control plane reports for a live machine.
LIVE_STATES = ("started", "running")


def token_from_env() -> str:
    token = os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise ConfigError(
            f"cloud target needs an API key — set {TOKEN_ENV} (an smk_… tenant "
            "key, or `smol auth login`)"
        )
    return token


def base_url_from_env() -> str:
    return (os.environ.get(URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


class CloudClient:
    """Thread-safe (stateless per request) client for /v1."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or token_from_env()
        self.base_url = (base_url or base_url_from_env()).rstrip("/")

    # ------------------------------------------------------------ plumbing

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        raw_body: bytes | None = None,
        timeout: float = API_TIMEOUT,
    ) -> Any:
        headers = {"authorization": f"Bearer {self.api_key}"}
        data = None
        if json_body is not None:
            headers["content-type"] = "application/json"
            data = json.dumps(json_body).encode()
        elif raw_body is not None:
            headers["content-type"] = "application/octet-stream"
            data = raw_body
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
                if not payload or resp.status == 204:
                    return None
                if "application/json" in (resp.headers.get("content-type") or ""):
                    return json.loads(payload)
                return payload
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode(errors="replace").strip()
            except Exception:  # noqa: BLE001 — error body is best-effort
                pass
            raise CloudError(exc.code, f"{method} {path}: {detail or exc.reason}") from None
        except urllib.error.URLError as exc:
            raise CloudError(0, f"{method} {path}: {getattr(exc, 'reason', exc)}") from None
        except TimeoutError:
            raise CloudError(0, f"{method} {path}: timed out after {timeout:.0f}s") from None

    # ------------------------------------------------------------ machines

    def create_machine(
        self,
        *,
        name: str,
        image: str,
        cpus: int,
        memory_mb: int,
        guest_ports: list[int],
        env: dict[str, str] | None = None,
        network: dict | None = None,
        disk_gb: int | None = None,
        forkable: bool = False,
        ttl_seconds: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": name,
            "source": {"type": "image", "reference": image},
            "resources": {"cpus": cpus, "memoryMb": memory_mb, "diskGb": disk_gb},
            "env": env or {},
            "ports": [{"port": p} for p in guest_ports],
            "ttlSeconds": ttl_seconds,
        }
        if network is not None:
            body["network"] = network
        if forkable:
            body["forkable"] = True
        return self.request("POST", "/v1/machines", json_body=body)

    def get_machine(self, machine_id: str) -> dict[str, Any]:
        return self.request("GET", f"/v1/machines/{machine_id}")

    def list_machines(self) -> list[dict[str, Any]]:
        listed = self.request("GET", "/v1/machines")
        if isinstance(listed, dict):
            listed = listed.get("machines", [])
        return [m for m in (listed or []) if isinstance(m, dict) and m.get("id")]

    def start_machine(self, machine_id: str) -> None:
        self.request("POST", f"/v1/machines/{machine_id}/start")

    def fork_machine(self, golden_id: str, name: str,
                     guest_ports: list[int] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if guest_ports:
            body["ports"] = [{"port": p} for p in guest_ports]
        return self.request(
            "POST", f"/v1/machines/{golden_id}/fork", json_body=body,
            timeout=120.0,  # forks serialize per golden and can queue
        )

    def stop_machine(self, machine_id: str) -> None:
        self.request("POST", f"/v1/machines/{machine_id}/stop")

    def delete_machine(self, machine_id: str) -> None:
        self.request("DELETE", f"/v1/machines/{machine_id}")

    def exec(
        self,
        machine_id: str,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        workdir: str | None = None,
        timeout: float | None = None,
        stdin: str | None = None,
    ) -> dict[str, Any]:
        """Returns the raw response: exitCode / stdout / stderr / *Truncated."""
        body: dict[str, Any] = {
            "command": command,
            "env": env or {},
            "cwd": workdir,
            "timeoutSeconds": int(timeout) if timeout else None,
        }
        if stdin is not None:
            body["stdin"] = stdin
        http_timeout = API_TIMEOUT if timeout is None else max(
            API_TIMEOUT, timeout + EXEC_HTTP_HEADROOM
        )
        return self.request(
            "POST", f"/v1/machines/{machine_id}/exec",
            json_body=body, timeout=http_timeout,
        )

    def upload_file(self, machine_id: str, guest_path: str, data: bytes) -> None:
        encoded = "/".join(quote(seg, safe="") for seg in guest_path.split("/"))
        self.request(
            "PUT", f"/v1/machines/{machine_id}/files/{encoded}",
            raw_body=data, timeout=max(API_TIMEOUT, len(data) / (256 * 1024)),
        )

    # ------------------------------------------------------------ waiting

    def wait_ready(self, machine_id: str, deadline_s: float = READY_DEADLINE) -> dict[str, Any]:
        start = time.monotonic()
        deadline = start + deadline_s
        info: dict[str, Any] = {}
        last_exc: CloudError | None = None
        while time.monotonic() < deadline:
            try:
                info = self.get_machine(machine_id)
                last_exc = None
            except CloudError as exc:
                # Only transport errors and 5xx are worth waiting out. Auth
                # failures are permanent, and a 404 is too once the machine
                # has had NOT_FOUND_GRACE to appear in the control plane.
                if exc.status in (401, 403):
                    raise
                if exc.status == 404 and time.monotonic() - start > NOT_FOUND_GRACE:
                    raise
                last_exc = exc
                info = {}
            state = info.get("state")
            if state in LIVE_STATES:
                return info
            if state == "error":
                raise CloudError(0, f"machine {machine_id} entered error state while starting")
            time.sleep(1.0)
        detail = f"state={info.get('state')}"
        if last_exc is not None:
            detail += f"; last error: {last_exc}"
        raise CloudError(
            0, f"machine {machine_id} not ready after {deadline_s:.0f}s ({detail})"
        )

    def wait_url(self, machine_id: str, deadline_s: float = URL_DEADLINE) -> str:
        """The ingress URL is provisioned asynchronously after start (usually
        a couple of seconds); poll until it appears."""
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            info = self.get_machine(machine_id)
            url = info.get("url")
            if url:
                return str(url)
            time.sleep(1.0)
        raise CloudError(
            0, f"machine {machine_id}: no ingress URL after {deadline_s:.0f}s — "
               "was it created with a published port?"
        )
