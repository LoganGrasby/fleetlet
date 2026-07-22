"""fleetlet exception types."""

from __future__ import annotations


class FleetletError(Exception):
    """Base class for all fleetlet errors."""


class ConfigError(FleetletError):
    """Invalid configuration (bad image/function/pool options)."""


class SmolvmError(FleetletError):
    """A smolvm CLI command failed."""

    def __init__(self, cmd: list[str], returncode: int, stderr: str = ""):
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr.strip()
        msg = f"`{' '.join(cmd)}` exited {returncode}"
        if self.stderr:
            msg += f"\n{self.stderr}"
        super().__init__(msg)


class CloudError(FleetletError):
    """A smol cloud API request failed (HTTP status, or 0 for transport)."""

    def __init__(self, status: int, message: str):
        self.status = status
        prefix = f"cloud API {status}: " if status else "cloud API: "
        super().__init__(prefix + message)


class WorkerError(FleetletError):
    """A worker VM died, failed to boot, or dropped its connection."""


class RemoteError(FleetletError):
    """The user function raised inside the worker VM.

    Carries the remote traceback. If the original exception could be
    transported back, it is attached as ``__cause__``.
    """

    def __init__(self, message: str, remote_traceback: str = ""):
        self.remote_traceback = remote_traceback
        if remote_traceback:
            message = f"{message}\n\n--- remote traceback ---\n{remote_traceback.rstrip()}"
        super().__init__(message)


class AppNotRunning(FleetletError):
    """A remote call was made outside of `with app.run():`."""


class FrozenActorError(FleetletError):
    """A method was called on an actor whose VM is frozen.

    ``actor.fork()`` freezes the actor's VM as the immutable branch
    template — call methods on a branch, or ``fork()`` again for more
    branches at the frozen state."""
