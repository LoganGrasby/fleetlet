"""User-facing handles: Function, Cls, ClsInstance, BoundMethod.

Also owns "spec resolution" — deciding how a callable travels to the guest:

* ``import``   — importable module function: ships as (module, qualname),
                 version-safe, the guest imports the synced project source.
* ``mainfile`` — defined in the user's entrypoint script (`__main__`): ships
                 as (relpath, qualname); the guest loads the file and aliases
                 it as __main__ so pickled __main__.X references resolve.
* ``blob``     — lambdas/closures/REPL functions: cloudpickle by value.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, TYPE_CHECKING

import cloudpickle

from ._proto import blob_id
from .errors import ConfigError
from .image import Image

if TYPE_CHECKING:
    from .app import App

DEFAULT_MAP_WORKERS = 4

# `fleetlet serve` loads the user's script under this module name (NOT
# "__main__", so demo `if __name__ == "__main__":` blocks don't fire).
# Functions defined there must still ship as mainfile specs.
SERVE_RUN_NAME = "__fleetlet_serve__"


def enter(fn: Callable | None = None):
    """Mark a method of an `@app.cls` class to run once at worker boot
    (before any calls; inside the golden for fork pools, so every clone
    inherits its effects). Usable as `@enter` or `@enter()`."""

    def mark(f: Callable) -> Callable:
        f.__fleetlet_enter__ = True  # type: ignore[attr-defined]
        return f

    return mark(fn) if fn is not None else mark


def resolve_spec(raw: Callable, project_root: str | None) -> tuple[dict[str, Any], bytes | None]:
    module = getattr(raw, "__module__", None)
    qualname = getattr(raw, "__qualname__", "") or getattr(raw, "__name__", "")

    if module and "<locals>" not in qualname and "<lambda>" not in qualname:
        if module not in ("__main__", SERVE_RUN_NAME):
            return {"kind": "import", "module": module, "qualname": qualname}, None
        rel = defining_file_rel(raw, project_root)
        if rel is not None:
            return {"kind": "mainfile", "file": rel, "qualname": qualname}, None

    blob = cloudpickle.dumps(raw)
    return {"kind": "blob", "blob_id": blob_id(blob)}, blob


def defining_file_rel(raw: Callable, project_root: str | None) -> str | None:
    """Relative path of the file that defines `raw`, if inside the project.
    Sourced from the code object — correct under both `python script.py`
    (module __main__) and `fleetlet serve script.py` (runpy, SERVE_RUN_NAME)."""
    code = getattr(raw, "__code__", None)
    file = getattr(code, "co_filename", None)
    return _rel_to_project(file, project_root)


def main_file_rel(project_root: str | None) -> str | None:
    """Relative path of the running script, if it lives inside the project."""
    main = sys.modules.get("__main__")
    return _rel_to_project(getattr(main, "__file__", None), project_root)


def _rel_to_project(file: str | None, project_root: str | None) -> str | None:
    if not file or not project_root or not os.path.exists(file):
        return None
    rel = os.path.relpath(os.path.abspath(file), os.path.abspath(project_root))
    if rel.startswith(".."):
        return None
    return rel


def slugify(name: str) -> str:
    """ASCII-only slug — these end up in smolvm machine names."""
    slug = "".join(
        c if ("a" <= c <= "z" or "0" <= c <= "9") else "-" for c in name.lower()
    ).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "fn"


@dataclass
class Options:
    """Per-function/cls pool options (decorator keyword arguments)."""

    image: Image | None = None
    cpus: int = 2
    memory: int = 1024
    gpu: bool = False
    gpu_vram: int | None = None
    cuda: bool = False
    share_weights: bool = False
    net: bool | list[str] = False
    volumes: dict[str, str] | list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    workers: int | None = None  # None = 1, but .map() may autoscale
    pool: str = "auto"  # auto | fork | cold
    timeout: float | None = None
    retries: int = 0
    name: str | None = None

    def normalized_volumes(self) -> list[str]:
        if not self.volumes:
            return []
        if isinstance(self.volumes, dict):
            return [f"{host}:{guest}" for host, guest in self.volumes.items()]
        return list(self.volumes)


class Function:
    """Handle returned by `@app.function`. Call via .remote/.spawn/.map/.local."""

    def __init__(self, app: "App", raw: Callable, options: Options):
        self._app = app
        self._raw = raw
        self._options = options
        self._slug = slugify(options.name or getattr(raw, "__name__", "fn"))
        self._cached_spec: tuple[dict[str, Any], bytes | None] | None = None
        self.__name__ = getattr(raw, "__name__", self._slug)
        self.__doc__ = getattr(raw, "__doc__", None)

    # Guest-side unwrap hook: the runner resolves the decorated symbol and
    # takes this attribute to reach the undecorated callable.
    @property
    def _fleetlet_raw(self) -> Callable:
        return self._raw

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise ConfigError(
            f"'{self.__name__}' is a fleetlet Function — use "
            f".remote(...) to run it in a VM or .local(...) to run it here"
        )

    def local(self, *args: Any, **kwargs: Any) -> Any:
        return self._raw(*args, **kwargs)

    def spawn(self, *args: Any, **kwargs: Any) -> cf.Future:
        return self._submit(args, kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._submit(args, kwargs).result()

    def map(
        self,
        *iterables: Iterable[Any],
        workers: int | None = None,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> Iterator[Any]:
        items = list(zip(*iterables))
        if not items:
            return iter(())
        size = workers or _map_pool_size(self._options, len(items))
        futures = [
            self._submit(item, {}, timeout=timeout, size_hint=size) for item in items
        ]
        return _iter_results(futures, return_exceptions)

    def _submit(self, args: tuple, kwargs: dict,
                timeout: float | None = None, size_hint: int | None = None) -> cf.Future:
        if self._cached_spec is None:
            self._cached_spec = resolve_spec(self._raw, self._app.project_root)
        spec, blob = self._cached_spec
        pool = self._app._pool_for_function(self)
        return pool.submit(
            spec, blob, args, kwargs,
            tag=self._slug, timeout=timeout, size_hint=size_hint,
        )


def _map_pool_size(options: Options, n_items: int) -> int:
    """An explicit workers= on the decorator is authoritative (including
    workers=1); otherwise autoscale to the item count, capped at
    DEFAULT_MAP_WORKERS."""
    if options.workers is not None:
        return options.workers
    return min(n_items, DEFAULT_MAP_WORKERS)


def _iter_results(futures: list[cf.Future], return_exceptions: bool) -> Iterator[Any]:
    for future in futures:
        try:
            yield future.result()
        except Exception as exc:
            if return_exceptions:
                yield exc
            else:
                raise


class Cls:
    """Handle returned by `@app.cls`. Instantiate to get a remote actor."""

    def __init__(self, app: "App", raw: type, options: Options):
        self._app = app
        self._raw = raw
        self._options = options
        self._slug = slugify(options.name or raw.__name__)
        self.__name__ = raw.__name__

    @property
    def _fleetlet_raw(self) -> type:
        return self._raw

    def __call__(self, *args: Any, **kwargs: Any) -> "ClsInstance":
        return ClsInstance(self, args, kwargs)


class ClsInstance:
    """A remote actor: one instance per worker, constructed + @enter'd at
    worker boot (in the golden, for fork pools). Method access returns
    BoundMethod handles."""

    def __init__(self, cls: Cls, args: tuple, kwargs: dict):
        self._cls = cls
        self._ctor_args = (args, kwargs)
        self._ctor_blob = cloudpickle.dumps((args, kwargs))
        ctor_hash = hashlib.sha256(self._ctor_blob).hexdigest()[:4]
        self._slug = f"{cls._slug}-{ctor_hash}"
        self._local_instance: Any | None = None

    def __getattr__(self, name: str) -> "BoundMethod":
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._cls._raw, name, None)
        if not callable(attr):
            raise AttributeError(
                f"{self._cls.__name__} has no method '{name}'"
            )
        return BoundMethod(self, name)

    def _local(self) -> Any:
        if self._local_instance is None:
            args, kwargs = self._ctor_args
            self._local_instance = self._cls._raw(*args, **kwargs)
            for klass in reversed(type(self._local_instance).__mro__):
                for attr_name, attr in vars(klass).items():
                    if getattr(attr, "__fleetlet_enter__", False):
                        getattr(self._local_instance, attr_name)()
        return self._local_instance


class BoundMethod:
    def __init__(self, instance: ClsInstance, method: str):
        self._instance = instance
        self._method = method

    def local(self, *args: Any, **kwargs: Any) -> Any:
        return getattr(self._instance._local(), self._method)(*args, **kwargs)

    def spawn(self, *args: Any, **kwargs: Any) -> cf.Future:
        return self._submit(args, kwargs)

    def remote(self, *args: Any, **kwargs: Any) -> Any:
        return self._submit(args, kwargs).result()

    def map(
        self,
        *iterables: Iterable[Any],
        workers: int | None = None,
        timeout: float | None = None,
        return_exceptions: bool = False,
    ) -> Iterator[Any]:
        items = list(zip(*iterables))
        if not items:
            return iter(())
        size = workers or _map_pool_size(self._instance._cls._options, len(items))
        futures = [
            self._submit(item, {}, timeout=timeout, size_hint=size) for item in items
        ]
        return _iter_results(futures, return_exceptions)

    def _submit(self, args: tuple, kwargs: dict,
                timeout: float | None = None, size_hint: int | None = None) -> cf.Future:
        instance = self._instance
        pool = instance._cls._app._pool_for_instance(instance)
        spec = {"kind": "self", "qualname": self._method}
        return pool.submit(
            spec, None, args, kwargs,
            tag=f"{instance._slug}.{self._method}",
            timeout=timeout, size_hint=size_hint,
        )
