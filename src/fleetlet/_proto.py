"""Host side of the fleetlet wire protocol.

Frames are `4-byte big-endian length + pickle(dict)`. The envelope dict never
contains user-defined types — user objects travel as nested `bytes` blobs so
the guest can defer unpickling them until `__main__` aliasing is in place.

The guest half of this protocol lives in `_runner.py`, which is shipped into
the VM as a standalone file and therefore duplicates the ~15 framing lines.
Keep `PROTO_VERSION` in sync between the two files.
"""

from __future__ import annotations

import hashlib
import pickle
import socket
import struct
from typing import Any

PROTO_VERSION = 1
PICKLE_PROTOCOL = 5
GUEST_PORT = 7777
GUEST_ROOT = "/opt/fleetlet"
GUEST_PROJECT = f"{GUEST_ROOT}/project"
GUEST_RUNNER = f"{GUEST_ROOT}/runner.py"
# On the persistent overlay, NOT /tmp: `machine exec` gets a fresh tmpfs view
# per call, so a /tmp log written by the detached runner is invisible to a
# separate `exec ... cat`. The overlay is shared across execs.
GUEST_LOG = f"{GUEST_ROOT}/runner.log"


def send_frame(sock: socket.socket, obj: dict[str, Any]) -> None:
    data = pickle.dumps(obj, protocol=PICKLE_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_frame(sock: socket.socket) -> dict[str, Any] | None:
    """Read one frame; None on clean EOF."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    body = _recv_exact(sock, length)
    if body is None:
        return None
    return pickle.loads(body)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None if not buf else _short_read(len(buf), n)
        buf.extend(chunk)
    return bytes(buf)


def _short_read(got: int, want: int) -> bytes:
    raise ConnectionError(f"short read: got {got} of {want} bytes")


def blob_id(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()[:16]
