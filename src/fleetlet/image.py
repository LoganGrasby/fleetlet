"""Image definitions — Modal-style chained builders.

An Image is a base OCI tag plus an ordered list of shell build steps that are
"baked" into a worker (or fork golden) right after first boot. Instances are
immutable: every chained method returns a new Image.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import sys
from dataclasses import dataclass, field, replace
from typing import Mapping


@dataclass(frozen=True)
class Image:
    base: str
    steps: tuple[str, ...] = ()
    env_vars: tuple[tuple[str, str], ...] = ()
    workdir_path: str | None = None

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_registry(cls, tag: str) -> "Image":
        """An image straight from a registry, e.g. `python:3.12-slim`."""
        return cls(base=tag)

    @classmethod
    def from_smolmachine(cls, path: str) -> "Image":
        """A packed `.smolmachine` artifact (from `smolvm pack create`).
        Boots from pre-extracted layers — no registry pull, works offline."""
        return cls(base=os.path.abspath(path))

    @classmethod
    def default(cls) -> "Image":
        """`python:X.Y-slim` matching the host interpreter's minor version,
        so pickled code objects (lambdas, closures) stay compatible.

        `FLEETLET_DEFAULT_IMAGE` overrides this (a registry tag or a
        `.smolmachine` path) — for mirrors, air-gapped hosts, or CI."""
        override = os.environ.get("FLEETLET_DEFAULT_IMAGE")
        if override:
            if override.endswith(".smolmachine"):
                return cls.from_smolmachine(override)
            return cls(base=override)
        major, minor = sys.version_info[:2]
        return cls(base=f"python:{major}.{minor}-slim")

    # ------------------------------------------------------------ build steps

    def run_commands(self, *commands: str) -> "Image":
        return replace(self, steps=self.steps + tuple(commands))

    def pip_install(self, *packages: str) -> "Image":
        if not packages:
            return self
        quoted = " ".join(shlex.quote(p) for p in packages)
        return self.run_commands(f"python3 -m pip install --no-input -q {quoted}")

    def apt_install(self, *packages: str) -> "Image":
        if not packages:
            return self
        quoted = " ".join(shlex.quote(p) for p in packages)
        return self.run_commands(
            f"apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -yqq {quoted}"
        )

    def apk_add(self, *packages: str) -> "Image":
        if not packages:
            return self
        quoted = " ".join(shlex.quote(p) for p in packages)
        return self.run_commands(f"apk add --no-cache -q {quoted}")

    # ------------------------------------------------------------ config

    def env(self, vars: Mapping[str, str] | None = None, **kwargs: str) -> "Image":
        merged = dict(self.env_vars)
        merged.update(vars or {})
        merged.update(kwargs)
        return replace(self, env_vars=tuple(sorted(merged.items())))

    def workdir(self, path: str) -> "Image":
        return replace(self, workdir_path=path)

    # ------------------------------------------------------------ introspection

    @property
    def needs_network(self) -> bool:
        """Build steps run inside the guest, so any step needs egress."""
        return bool(self.steps)

    @property
    def content_id(self) -> str:
        payload = json.dumps(
            [self.base, list(self.steps), list(self.env_vars), self.workdir_path]
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Image({self.base!r}, steps={len(self.steps)})"
