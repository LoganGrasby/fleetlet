"""`fleetlet deploy` — run an app's HTTP gateway persistently on the cloud.

The Modal-deploy analog on smol cloud machines: one cloud microVM is
created with the serve port published, the fleetlet runtime + your project
are staged into it, and `python3 -m fleetlet serve <script> --local` is
launched as a long-lived process. The platform's ingress gives it a stable
HTTPS URL, reachable with your tenant API key:

    curl -H "Authorization: Bearer $SMOL_CLOUD_TOKEN" \\
         -X POST https://<name>-<tenant>.apps.smolmachines.com/call/embed \\
         -d '{"text": "hi"}'

`--local` inside the VM means function calls run in the gateway's own
process — the whole service is one hardware-isolated microVM. (Deployed
gateways spawning their own per-function worker VMs is a natural follow-up;
it only needs the tenant key + egress passed into the VM.)

Unlike pool workers, deployments get NO TTL — they live until
`fleetlet clean --cloud --deployments` (or an API delete) removes them.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from ._backend import get_backend
from ._function import slugify
from ._log import log
from ._machine import MachineSpec
from ._pool import _probe_python3
from ._proto import GUEST_LOG, GUEST_PROJECT, GUEST_ROOT
from .errors import CloudError, ConfigError

DEPLOY_PREFIX = "flt-deploy-"
HEALTH_DEADLINE = 90.0


@dataclass
class Deployment:
    name: str          # machine name (flt-deploy-…)
    url: str           # ingress base URL
    port: int


def deploy(
    app,
    script_path: str,
    *,
    name: str | None = None,
    port: int = 8283,
    cpus: int = 1,
    memory_mib: int = 512,
    net: bool = False,
) -> Deployment:
    """Create (or replace) a cloud machine serving `app` from `script_path`."""
    backend = get_backend("cloud")

    if not app._functions:
        raise ConfigError("app has no @app.function definitions to deploy")
    if app.project_root is None:
        raise ConfigError("deploy needs a project to ship — don't use sync_project=False")
    script_rel = os.path.relpath(os.path.abspath(script_path), app.project_root)
    if script_rel.startswith(".."):
        raise ConfigError(
            f"{script_path} is outside the project root {app.project_root} — "
            "deploy ships the project directory, so the script must live in it"
        )
    image = app.image
    if image.base.endswith(".smolmachine"):
        raise ConfigError(
            "packed .smolmachine artifacts are local files — deploy needs a "
            "registry image (e.g. python:3.12-slim)"
        )
    if image.needs_network and not net:
        raise ConfigError(
            "the app image has build steps (pip/apt/run_commands), which need "
            "egress in the VM — pass --net"
        )

    machine = DEPLOY_PREFIX + slugify(name or app.name)

    # Redeploy = replace. Brief downtime; rolling deploys are the platform's
    # /v1/apps layer and can come later.
    for m in backend.list_machines():
        if m["name"] == machine:
            log(f"replacing existing deployment {machine}")
            backend.stop(m["name"])
            if not backend.delete(m["name"]):
                raise CloudError(
                    0,
                    f"could not delete the existing deployment {machine} — "
                    "aborting before creating a same-named machine (the old "
                    "one may now be stopped; retry, or remove it with "
                    "`fleetlet clean --cloud --deployments`)",
                )

    log(f"deploying '{app.name}' as {machine} ({image.base}, port {port})…")
    t0 = time.monotonic()
    try:
        # create sits inside the try: if the POST lands server-side but the
        # response is lost, the handler below still stops+deletes by name —
        # deployments have no TTL, so a leak here would bill forever.
        backend.create(MachineSpec(
            name=machine,
            image=image.base,
            cpus=cpus,
            memory_mib=memory_mib,
            ports=[(0, port)],
            env={},
            net=net,
            ttl_seconds=0,  # deployments are persistent — no safety-net TTL
        ))
        backend.start(machine)

        if not _probe_python3(backend, machine):
            raise ConfigError(
                f"image '{image.base}' has no python3 — use a Python image"
            )
        backend.execute(machine, ["sh", "-c", f"mkdir -p {GUEST_ROOT} {GUEST_PROJECT}"])
        backend.put_file(machine, app._ensure_stage_tar(), f"{GUEST_ROOT}/stage.tgz")
        backend.execute(machine, ["sh", "-c",
                                  f"tar -xzf {GUEST_ROOT}/stage.tgz -C {GUEST_ROOT} "
                                  f"&& rm {GUEST_ROOT}/stage.tgz"])
        backend.put_file(machine, app._ensure_project_tar(), f"{GUEST_ROOT}/project.tgz")
        backend.execute(machine, ["sh", "-c",
                                  f"tar -xzf {GUEST_ROOT}/project.tgz -C {GUEST_PROJECT} "
                                  f"&& rm {GUEST_ROOT}/project.tgz"])

        env = dict(image.env_vars)
        env["PYTHONPATH"] = GUEST_ROOT
        for i, step in enumerate(image.steps):
            log(f"bake {i + 1}/{len(image.steps)}: {step[:60]}")
            backend.execute(machine, ["sh", "-c", step], env=env)

        backend.execute(
            machine,
            ["python3", "-m", "fleetlet", "serve", f"{GUEST_PROJECT}/{script_rel}",
             "--local", "--host", "0.0.0.0", "--port", str(port)],
            env=env, workdir=GUEST_PROJECT, detach=True,
        )

        url = backend.ingress_url(machine).rstrip("/")
        _wait_healthy(url, backend.api_key, machine, backend)
    except BaseException:
        # A machine that never became a service shouldn't linger (and bill).
        backend.stop(machine)
        backend.delete(machine)
        raise
    log(f"deployed {machine} in {time.monotonic() - t0:.1f}s")
    return Deployment(name=machine, url=url, port=port)


def _wait_healthy(url: str, api_key: str, machine: str, backend) -> None:
    deadline = time.monotonic() + HEALTH_DEADLINE
    last: object = None
    while time.monotonic() < deadline:
        req = urllib.request.Request(
            url + "/health", headers={"authorization": f"Bearer {api_key}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    return
                last = resp.status
        except urllib.error.HTTPError as exc:
            last = exc.code
        except (OSError, urllib.error.URLError) as exc:
            last = exc
        time.sleep(1.5)
    # Surface the gateway's own log — the failure reason lives in the guest.
    tail = ""
    try:
        proc = backend.execute(machine, ["sh", "-c", f"tail -40 {GUEST_LOG}"], check=False)
        tail = proc.stdout.decode(errors="replace").strip()
    except CloudError:
        pass
    raise ConfigError(
        f"deployment {machine} never became healthy at {url}/health "
        f"(last: {last!r})" + (f"\n--- guest log ---\n{tail}" if tail else "")
    )
