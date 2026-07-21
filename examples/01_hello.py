"""Hello fleetlet: run a function in a hardware-isolated microVM.

    python examples/01_hello.py
    FLEETLET_TARGET=cloud python examples/01_hello.py   # same, on a cloud VM
"""

import platform

import fleetlet

app = fleetlet.App("hello")


@app.function
def where_am_i(greeting: str) -> str:
    import os

    return (
        f"{greeting}! I ran on {platform.system()} {platform.machine()} "
        f"(pid {os.getpid()}, host {platform.node()})"
    )


if __name__ == "__main__":
    print(f"host is {platform.system()} {platform.machine()}")
    with app.run():
        print(where_am_i.remote("hi"))
