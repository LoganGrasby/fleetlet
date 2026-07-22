"""The App: registry of functions/classes and owner of worker pools."""

from __future__ import annotations

import atexit
import concurrent.futures as cf
import dataclasses
import hashlib
import os
import secrets
import tarfile
import tempfile
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from ._backend import Backend, get_backend, resolve_target
from ._function import Cls, ClsInstance, Function, Options, main_file_rel, resolve_spec, slugify
from ._log import log
from ._pool import Pool, PoolConfig, build_stage_tar
from .errors import AppNotRunning, ConfigError
from .image import Image

EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", ".tox", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode",
    ".fleetlet", "dist", "build", ".eggs",
}
PROJECT_WARN_MB = 64
PROJECT_MAX_MB = 256


class App:
    def __init__(
        self,
        name: str = "app",
        *,
        image: Image | None = None,
        project_root: str | None = None,
        project_ignore: list[str] | None = None,
        sync_project: bool = True,
        target: str | None = None,
    ):
        """``target`` picks where worker VMs run: ``"local"`` (the smolvm
        engine on this machine, default) or ``"cloud"`` (the smol cloud —
        needs SMOL_CLOUD_TOKEN). Unset, it follows $FLEETLET_TARGET."""
        self.name = slugify(name)
        self.image = image or Image.default()
        self.project_root = os.path.abspath(project_root or os.getcwd()) if sync_project else None
        self.project_ignore = set(project_ignore or [])
        self.target = resolve_target(target)
        self._backend: Backend | None = None
        self.run_id: str | None = None
        self._functions: dict[str, Function] = {}
        self._classes: dict[str, Cls] = {}
        self._pools: dict[str, Pool] = {}
        self._pool_lock = threading.Lock()
        self._project_tar: str | None = None
        self._stage_tar: str | None = None
        self._atexit_registered = False

    # ------------------------------------------------------------ decorators

    def function(self, fn: Callable | None = None, **kwargs: Any):
        """Register a function to run in worker VMs.

        Keyword args (all optional): image, cpus, memory (MiB), gpu, gpu_vram,
        cuda, share_weights, net, volumes, env, workers, pool ("auto"|"fork"|
        "cold"), timeout, retries, name.
        """
        options = Options(**kwargs)

        def wrap(f: Callable) -> Function:
            handle = Function(self, f, options)
            if handle._slug in self._functions:
                raise ConfigError(
                    f"duplicate function name '{handle._slug}' — pass name= to disambiguate"
                )
            self._functions[handle._slug] = handle
            return handle

        return wrap(fn) if fn is not None else wrap

    def cls(self, klass: type | None = None, **kwargs: Any):
        """Register a class as a remote actor. One instance lives per worker;
        methods marked @fleetlet.enter run at worker boot (in the golden for
        fork pools — clones inherit the warmed state). With forkable=True the
        live instance can be branched at runtime via instance.fork()."""
        options = Options(**kwargs)

        def wrap(k: type) -> Cls:
            handle = Cls(self, k, options)
            if handle._slug in self._classes:
                raise ConfigError(
                    f"duplicate cls name '{handle._slug}' — pass name= to disambiguate"
                )
            self._classes[handle._slug] = handle
            return handle

        return wrap(klass) if klass is not None else wrap

    # ------------------------------------------------------------ run context

    @contextmanager
    def run(self) -> Iterator["App"]:
        """Start the app: pools spin up lazily on first call, and everything
        is torn down (VMs stopped + deleted) on exit."""
        if self.run_id is not None:
            raise ConfigError("app is already running")
        self.run_id = secrets.token_hex(3)
        if not self._atexit_registered:
            atexit.register(self._teardown)
            self._atexit_registered = True
        where = " on the cloud" if self.target == "cloud" else ""
        log(f"app '{self.name}' running{where} (run id {self.run_id})")
        try:
            yield self
        finally:
            self._teardown()
            self.run_id = None

    def serve(self, host: str = "127.0.0.1", port: int = 8283,
              *, warm: bool = False, local: bool = False) -> None:
        """Serve this app's functions over HTTP (see fleetlet/serve.py).
        Blocks until interrupted; VMs are torn down on exit."""
        from .serve import serve as _serve

        _serve(self, host, port, warm=warm, local=local)

    def _teardown(self) -> None:
        with self._pool_lock:
            pools, self._pools = list(self._pools.values()), {}
        if pools:
            with cf.ThreadPoolExecutor(max_workers=len(pools)) as executor:
                list(executor.map(lambda p: p.shutdown(), pools))
        for attr in ("_project_tar", "_stage_tar"):
            path = getattr(self, attr)
            if path and os.path.exists(path):
                os.unlink(path)
            setattr(self, attr, None)

    # ------------------------------------------------------------ pools

    def _pool_for_function(self, fn: Function) -> Pool:
        setup_seed = f"fn:{fn._slug}"
        return self._get_pool(fn._slug, fn._options, setup_seed,
                              cls_spec=None, cls_blob=None, cls_args=None)

    def _pool_for_instance(self, instance: ClsInstance) -> Pool:
        cls = instance._cls
        cls_spec, cls_blob = resolve_spec(cls._raw, self.project_root)
        setup_seed = f"cls:{instance._slug}:{cls_spec}:{instance._ctor_blob.hex()[:32]}"
        return self._get_pool(instance._slug, cls._options, setup_seed,
                              cls_spec=cls_spec, cls_blob=cls_blob,
                              cls_args=instance._ctor_blob)

    def _get_pool(
        self,
        slug: str,
        options: Options,
        setup_seed: str,
        *,
        cls_spec: dict[str, Any] | None,
        cls_blob: bytes | None,
        cls_args: bytes | None,
    ) -> Pool:
        if self.run_id is None:
            raise AppNotRunning(
                f"cannot call '{slug}' — wrap remote calls in `with app.run():`"
            )
        with self._pool_lock:
            if slug in self._pools:
                return self._pools[slug]
            image = options.image or self.image
            setup_id = hashlib.sha256(
                f"{self.run_id}:{image.content_id}:{setup_seed}".encode()
            ).hexdigest()[:16]
            cfg = PoolConfig(
                app_name=self.name,
                fn_slug=slug,
                run_id=self.run_id,
                image=image,
                mode=options.pool,
                size=1 if options.workers is None else options.workers,
                cpus=options.cpus,
                memory_mib=options.memory,
                gpu=options.gpu,
                gpu_vram_mib=options.gpu_vram,
                cuda=options.cuda,
                share_weights=options.share_weights,
                net=options.net,
                volumes=options.normalized_volumes(),
                env=dict(options.env),
                project_tar=self._ensure_project_tar(),
                stage_tar=self._ensure_stage_tar(),
                main_file_rel=main_file_rel(self.project_root),
                cls_spec=cls_spec,
                cls_blob=cls_blob,
                cls_args=cls_args,
                setup_id=setup_id,
                call_timeout=options.timeout,
                retries=options.retries,
                forkable=options.forkable,
            )
            pool = Pool(cfg, self.backend)
            self._pools[slug] = pool
            return pool

    def _fork_instance_pools(self, instance: ClsInstance,
                             branch_slugs: list[str]) -> list[Pool]:
        """Fork a live actor's VM into one adopted pool per branch slug.

        The parent pool serializes the fork against in-flight calls and
        freezes on first use; each branch machine is then adopted into its
        own single-worker pool, registered so app teardown sweeps it."""
        pool = self._pool_for_instance(instance)
        branch_cfgs = [
            dataclasses.replace(pool.cfg, fn_slug=slug, forkable=False)
            for slug in branch_slugs
        ]
        forked = pool.fork_branches(
            [f"{cfg.machine_prefix()}-w0" for cfg in branch_cfgs]
        )
        sent_blobs = set(pool.workers[0].sent_blobs) if pool.workers else set()
        adopted: list[Pool] = []
        failures: list[Exception] = []
        with cf.ThreadPoolExecutor(max_workers=len(forked)) as executor:
            futures = {
                executor.submit(Pool.adopt, cfg, self.backend, name, port,
                                sent_blobs): name
                for cfg, (name, port) in zip(branch_cfgs, forked)
            }
            for future, name in futures.items():
                try:
                    adopted.append(future.result())
                except Exception as exc:
                    failures.append(exc)
                    # No pool owns this machine yet, so no teardown sweep
                    # will ever find it — reclaim it here.
                    self.backend.stop(name)
                    self.backend.delete(name)
        with self._pool_lock:
            for branch_pool in adopted:
                self._pools[branch_pool.cfg.fn_slug] = branch_pool
        if failures:
            for extra in failures[1:]:
                log(f"additional branch adoption failure: {extra}")
            raise failures[0]
        return adopted

    @property
    def backend(self) -> Backend:
        """The machine backend for this app's target, built on first use (the
        cloud one needs SMOL_CLOUD_TOKEN, so don't demand it at import)."""
        if self._backend is None:
            self._backend = get_backend(self.target)
        return self._backend

    # ------------------------------------------------------------ project sync

    def _ensure_stage_tar(self) -> str:
        if self._stage_tar is None:
            self._stage_tar = build_stage_tar()
        return self._stage_tar

    def _ensure_project_tar(self) -> str | None:
        if self.project_root is None:
            return None
        if self._project_tar is not None:
            return self._project_tar
        fd, path = tempfile.mkstemp(prefix="fleetlet-project-", suffix=".tgz")
        os.close(fd)
        total = 0
        with tarfile.open(path, "w:gz") as tar:
            for dirpath, dirnames, filenames in os.walk(self.project_root):
                dirnames[:] = [
                    d for d in dirnames
                    if d not in EXCLUDE_DIRS and d not in self.project_ignore
                ]
                for filename in filenames:
                    if filename.endswith((".pyc", ".pyo")) or filename in self.project_ignore:
                        continue
                    full = os.path.join(dirpath, filename)
                    if not os.path.isfile(full):
                        continue
                    total += os.path.getsize(full)
                    if total > PROJECT_MAX_MB * 1024 * 1024:
                        os.unlink(path)
                        raise ConfigError(
                            f"project at {self.project_root} exceeds {PROJECT_MAX_MB}MB. "
                            "Set App(project_root=...) to a smaller directory, add "
                            "project_ignore entries, or App(sync_project=False)."
                        )
                    tar.add(full, arcname=os.path.relpath(full, self.project_root))
        if total > PROJECT_WARN_MB * 1024 * 1024:
            log(f"warning: syncing {total // (1024 * 1024)}MB of project files into "
                "each worker — consider App(project_root=...) or project_ignore")
        self._project_tar = path
        return path
