"""The `fleetlet` command: run scripts, serve/deploy apps, clean up leaks."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

from . import __version__
from . import _machine
from ._backend import TARGET_ENV, get_backend
from ._log import log
from .errors import FleetletError

PREFIX = "flt-"
DEPLOY_PREFIX = "flt-deploy-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fleetlet",
        description="Modal/Ray-style functions and actors on smolvm microVMs.",
    )
    parser.add_argument("--version", action="version", version=f"fleetlet {__version__}")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="run a fleetlet script (as __main__)")
    run_p.add_argument("script")
    run_p.add_argument("--cloud", action="store_true",
                       help="run worker VMs on the smol cloud (SMOL_CLOUD_TOKEN)")
    run_p.add_argument("args", nargs=argparse.REMAINDER)

    serve_p = sub.add_parser("serve", help="serve an app's functions over HTTP")
    serve_p.add_argument("script")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8283)
    serve_p.add_argument("--app", dest="app_var",
                         help="App variable name (if the script defines several)")
    serve_p.add_argument("--warm", action="store_true",
                         help="boot all pools before accepting requests")
    serve_p.add_argument("--local", action="store_true",
                         help="dev mode: run functions in-process, no VMs")
    serve_p.add_argument("--cloud", action="store_true",
                         help="workers on the smol cloud (gateway stays on this host)")

    deploy_p = sub.add_parser(
        "deploy", help="run the app's HTTP gateway persistently on the smol cloud")
    deploy_p.add_argument("script")
    deploy_p.add_argument("--name", help="deployment name (default: the app name)")
    deploy_p.add_argument("--port", type=int, default=8283)
    deploy_p.add_argument("--app", dest="app_var")
    deploy_p.add_argument("--cpus", type=int, default=1)
    deploy_p.add_argument("--memory", type=int, default=512, help="MiB")
    deploy_p.add_argument("--net", action="store_true",
                          help="give the service VM open egress")

    ls_p = sub.add_parser("ls", help="list fleetlet machines")
    ls_p.add_argument("--cloud", action="store_true", help="list cloud machines")

    clean_p = sub.add_parser("clean", help="stop + delete leaked worker machines")
    clean_p.add_argument("--app", help="only machines of this app (slug)")
    clean_p.add_argument("--cloud", action="store_true", help="clean cloud machines")
    clean_p.add_argument("--deployments", action="store_true",
                         help="also delete flt-deploy-* services (kept by default)")

    sub.add_parser("doctor", help="check the smolvm installation and cloud access")

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "deploy":
            return cmd_deploy(args)
        if args.command == "ls":
            return cmd_ls(args)
        if args.command == "clean":
            return cmd_clean(args)
        if args.command == "doctor":
            return cmd_doctor()
    except FleetletError as exc:
        # Auth/API/machine failures get one readable line, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.print_help()
    return 1


def _backend_for(cloud: bool):
    return get_backend("cloud" if cloud else "local")


def _fleet_machines(backend, app: str | None = None) -> list[dict]:
    if app is None:
        return [m for m in backend.list_machines() if m["name"].startswith(PREFIX)]
    # Workers embed a truncated app slug; deployments are `flt-deploy-<slug>`
    # with no suffix, so they must be matched by exact name.
    worker_prefix = f"{PREFIX}{app[:12]}-"
    deploy_name = f"{DEPLOY_PREFIX}{app}"
    return [m for m in backend.list_machines()
            if m["name"].startswith(worker_prefix) or m["name"] == deploy_name]


def _load_app(script: str, app_var: str | None):
    """Import the script (without firing its __main__ demo block) and return
    its App. Shared by serve and deploy."""
    from ._function import SERVE_RUN_NAME
    from .app import App

    namespace = runpy.run_path(script, run_name=SERVE_RUN_NAME)
    apps = {k: v for k, v in namespace.items() if isinstance(v, App)}
    if not apps:
        print(f"no fleetlet App found in {script}", file=sys.stderr)
        return None
    if app_var:
        if app_var not in apps:
            print(f"no App named '{app_var}' in {script} "
                  f"(found: {', '.join(apps)})", file=sys.stderr)
            return None
        return apps[app_var]
    if len(apps) == 1:
        return next(iter(apps.values()))
    print(f"multiple Apps in {script} ({', '.join(apps)}) — "
          "pick one with --app NAME", file=sys.stderr)
    return None


def cmd_run(args: argparse.Namespace) -> int:
    if args.cloud:
        os.environ[TARGET_ENV] = "cloud"
    sys.argv = [args.script, *args.args]
    runpy.run_path(args.script, run_name="__main__")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .serve import serve

    if args.cloud:
        os.environ[TARGET_ENV] = "cloud"  # before the script constructs its App
    app = _load_app(args.script, args.app_var)
    if app is None:
        return 1
    serve(app, host=args.host, port=args.port, warm=args.warm, local=args.local)
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    from ._deploy import deploy

    app = _load_app(args.script, args.app_var)
    if app is None:
        return 1
    dep = deploy(
        app, args.script,
        name=args.name, port=args.port,
        cpus=args.cpus, memory_mib=args.memory, net=args.net,
    )
    print(f"\n  {dep.url}")
    print("\nThe URL is private to your tenant — authenticate with your API key:")
    print(f"  curl -H \"Authorization: Bearer $SMOL_CLOUD_TOKEN\" {dep.url}/")
    fn = next(iter(app._functions))
    print(f"  curl -H \"Authorization: Bearer $SMOL_CLOUD_TOKEN\" "
          f"-X POST {dep.url}/call/{fn} -d '{{...}}'")
    slug = dep.name.removeprefix(DEPLOY_PREFIX)
    print(f"\nRemove with: fleetlet clean --cloud --deployments --app {slug}")
    return 0


def cmd_ls(args: argparse.Namespace) -> int:
    machines = _fleet_machines(_backend_for(args.cloud))
    if not machines:
        print("no fleetlet machines" + (" on the cloud" if args.cloud else ""))
        return 0
    for m in machines:
        line = f"{m['name']:<48} {m['state']}"
        if m.get("url"):
            line += f"  {m['url']}"
        print(line)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    backend = _backend_for(args.cloud)
    machines = _fleet_machines(backend, args.app)
    if not args.deployments:
        # Deployments are intentional long-lived services, not leaks.
        kept = [m for m in machines if m["name"].startswith(DEPLOY_PREFIX)]
        machines = [m for m in machines if not m["name"].startswith(DEPLOY_PREFIX)]
        if kept:
            log(f"keeping {len(kept)} deployment(s) — pass --deployments to remove")
    if not machines:
        print("nothing to clean")
        return 0
    # Stop running clones before frozen goldens, then delete everything.
    ordered = sorted(machines, key=lambda m: m["state"] == "frozen")
    for m in ordered:
        if m["state"] in ("running", "frozen"):
            log(f"stopping {m['name']}")
            backend.stop(m["name"])
    deleted = 0
    for m in ordered:
        log(f"deleting {m['name']}")
        if backend.delete(m["name"]):
            deleted += 1
    survivors = [
        s for s in _fleet_machines(backend, args.app)
        if any(s["name"] == m["name"] for m in machines)
    ]
    if survivors:
        print(f"cleaned {deleted} machine(s); {len(survivors)} would not delete:")
        for m in survivors:
            print(f"  {m['name']} ({m['state']})")
        return 1
    print(f"cleaned {deleted} machine(s)")
    return 0


def cmd_doctor() -> int:
    path = _machine.smolvm_available()
    if not path:
        print("✗ smolvm not found — install: curl -sSL https://smolmachines.com/install.sh | bash")
        return 1
    proc = _machine._run(["--version"], check=False)
    version = proc.stdout.decode().strip() or proc.stderr.decode().strip()
    print(f"✓ smolvm: {path} ({version})")
    print(f"✓ python: {sys.version.split()[0]}")
    try:
        import cloudpickle

        print(f"✓ cloudpickle: {cloudpickle.__version__}")
    except ImportError:
        print("✗ cloudpickle missing (pip install cloudpickle)")
        return 1
    _doctor_cloud()
    leaked = _fleet_machines(_backend_for(False))
    if leaked:
        print(f"! {len(leaked)} leftover machine(s) — run `fleetlet clean`")
    return 0


def _doctor_cloud() -> None:
    from ._cloud import TOKEN_ENV, CloudClient

    if not os.environ.get(TOKEN_ENV, "").strip():
        print(f"- cloud: no {TOKEN_ENV} set (local-only)")
        return
    try:
        n = len(CloudClient().list_machines())
        print(f"✓ cloud: authenticated ({n} machine(s) on the tenant)")
    except Exception as exc:  # noqa: BLE001 — doctor reports, never crashes
        print(f"✗ cloud: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
