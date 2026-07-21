"""Fan a workload out over a fork pool: one golden VM boots, N clones spawn
from it in ~100ms each and chew through the map in parallel.

    python examples/02_map.py
    FLEETLET_TARGET=cloud python examples/02_map.py   # same, on cloud VMs
"""

import time

import fleetlet

app = fleetlet.App("mapdemo")


@app.function(workers=4)  # workers > 1 → fork pool by default
def slow_square(x: int) -> int:
    import os
    import time as t

    t.sleep(1.0)  # pretend this is real work
    print(f"squared {x} in vm pid {os.getpid()}")
    return x * x


if __name__ == "__main__":
    with app.run():
        t0 = time.monotonic()
        results = list(slow_square.map(range(12)))
        elapsed = time.monotonic() - t0
        print(f"results: {results}")
        print(f"12 x 1s of work across 4 VMs in {elapsed:.1f}s")
