"""Each worker is a hardware-isolated VM with CoW state: crashes and file
writes stay contained, and workers get replaced automatically.

    python examples/04_isolation.py
    FLEETLET_TARGET=cloud python examples/04_isolation.py
"""

import fleetlet

app = fleetlet.App("isolation")


@app.function(workers=2, retries=1)
def hostile(n: int) -> str:
    import os

    # Scribble over the filesystem — each VM has its own CoW overlay.
    with open("/etc/passwd", "a") as f:
        f.write("hax:x:0:0::/:/bin/sh\n")
    if n == 3:
        os._exit(1)  # hard-kill the worker VM's runner mid-call
    return f"task {n}: wrote /etc/passwd in my own little world"


if __name__ == "__main__":
    with app.run():
        for result in hostile.map(range(6), return_exceptions=True):
            print(result)
    print("host /etc/passwd is untouched, dead worker was auto-replaced")
