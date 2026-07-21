"""Every @app.function doubles as an HTTP endpoint — route, parameter
schema, validation, and docs are generated from the signature, while calls execute in hardware-isolated microVMs.

    python examples/05_http_service.py                # serve on :8283 ($PORT overrides)
    fleetlet serve examples/05_http_service.py       # same, via CLI
    fleetlet serve examples/05_http_service.py --local   # dev, no VMs
    FLEETLET_TARGET=cloud python examples/05_http_service.py  # cloud workers

    curl -s localhost:8283/ | python3 -m json.tool
    curl -s -X POST localhost:8283/call/embed -d '{"text": "hello world"}'
    curl -s -X POST localhost:8283/call/shout -d '{"text": "hi", "times": 3}'
    curl -s -X POST localhost:8283/call/shout -d '{"times": "three"}'   # 400
"""

import os

import fleetlet

app = fleetlet.App("svc")


@app.function(workers=2)
def embed(text: str, dims: int = 8) -> list[float]:
    """Toy embedding: deterministic hash-derived vector."""
    import hashlib
    import struct

    digest = hashlib.sha256(text.encode()).digest()
    return [round(struct.unpack_from("<H", digest, i * 2)[0] / 65535, 4)
            for i in range(dims)]


@app.function()
def shout(text: str, times: int = 1) -> str:
    """Uppercase `text`, repeated `times` times."""
    return " ".join([text.upper()] * times)


if __name__ == "__main__":
    app.serve(port=int(os.environ.get("PORT", "8283")))
