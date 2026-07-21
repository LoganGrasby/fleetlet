"""Actors with expensive startup, the fork-pool flagship demo.

The @fleetlet.enter hook runs ONCE, inside the golden VM. Every worker is
then forked from that warmed golden — cloned memory and all — so the
"model load" cost is paid once no matter how many workers you scale to.

    python examples/03_actor.py
    FLEETLET_TARGET=cloud python examples/03_actor.py
"""

import time

import fleetlet

app = fleetlet.App("actordemo")


@app.cls(workers=3, pool="fork")
class Scorer:
    def __init__(self, offset: int = 0):
        self.offset = offset

    @fleetlet.enter
    def load_model(self):
        import os
        import time as t

        print(f"loading 'model' in pid {os.getpid()} (takes 5s, happens once)…")
        t.sleep(5)  # stand-in for torch.load / pipeline setup
        self.weights = [i * 0.5 for i in range(1000)]
        print("model loaded")

    def score(self, x: int) -> float:
        return self.weights[x % len(self.weights)] + self.offset


if __name__ == "__main__":
    with app.run():
        scorer = Scorer(offset=100)

        boot_story = "golden boot + ONE 5s model load + forks"
        t0 = time.monotonic()
        first = scorer.score.remote(10)
        print(f"first call: {first} (after {time.monotonic() - t0:.1f}s — includes {boot_story})")

        t0 = time.monotonic()
        results = list(scorer.score.map(range(30)))
        print(f"30 calls across 3 workers in {time.monotonic() - t0:.2f}s")
        print(f"sample: {results[:5]}")
