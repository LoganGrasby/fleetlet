"""Same code, cloud machines: flip the target and your functions run in
microVMs on the smol cloud instead of this machine. Nothing else changes —
`.remote()`, `.map()`, actors, crash-replacement all work identically.

    export SMOL_CLOUD_TOKEN=smk_...           # your tenant API key

    python examples/06_cloud.py               # target set in code below
    fleetlet run examples/01_hello.py --cloud   # or flip any script's target
    FLEETLET_TARGET=cloud python examples/02_map.py   # or via env

Deploy the HTTP gateway as a persistent cloud service (stable HTTPS URL,
tenant-authenticated by the platform):

    fleetlet deploy examples/05_http_service.py
    curl -H "Authorization: Bearer $SMOL_CLOUD_TOKEN" -X POST \\
         https://flt-deploy-svc-<tenant>.apps.smolmachines.com/call/embed \\
         -d '{"text": "hi"}'

Cloud differences to know about: fork-clone calls travel over the exec API
(clones get no ingress URL yet), workers carry a 1h server-side TTL as a
leak safety net (FLEETLET_CLOUD_TTL to change), and gpu/cuda/volumes are
local-only.
"""

import platform

import fleetlet

# target="cloud" here; omit it to follow $FLEETLET_TARGET (default local).
app = fleetlet.App("clouddemo", target="cloud")


@app.function(workers=3)
def crunch(n: int) -> str:
    import os

    return f"n={n} squared is {n * n} (pid {os.getpid()} on {platform.node()})"


if __name__ == "__main__":
    print(f"host: {platform.system()} {platform.machine()}")
    with app.run():
        print(crunch.remote(7))
        for line in crunch.map(range(9)):
            print(line)
