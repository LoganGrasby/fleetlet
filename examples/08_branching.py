"""Branch a LIVE actor: one warm state, many divergent futures.

`instance.fork(n)` snapshots a running actor's VM — interpreter, heap,
loaded data, everything — into n new actors in ~0.1s each. The branches
start from the exact same in-memory moment and then diverge, so expensive
state (datasets, models, seeded environments) is built once and explored
many ways in parallel.

Forking freezes the parent as an immutable template: further calls on it
raise FrozenActorError, while further fork() calls mint more branches at
the frozen state. (Local target only; branches can't re-fork.)

    python examples/08_branching.py
"""

import time

import fleetlet

app = fleetlet.App("branchdemo")


@app.cls(forkable=True)
class Trainer:
    @fleetlet.enter
    def load_dataset(self):
        import os
        import random

        print(f"loading dataset in pid {os.getpid()} (takes 3s, happens once)…")
        time.sleep(3)  # stand-in for download + parse
        random.seed(7)
        self.data = [random.gauss(0.0, 1.0) for _ in range(200_000)]
        self.history = [f"dataset loaded ({len(self.data):,} rows, pid {os.getpid()})"]

    def normalize(self) -> str:
        mean = sum(self.data) / len(self.data)
        self.data = [x - mean for x in self.data]
        self.history.append(f"normalized (mean was {mean:+.4f})")
        return self.history[-1]

    def train(self, lr: float, steps: int = 300) -> float:
        weight, loss = 0.0, float("inf")
        for step in range(steps):
            x = self.data[step % len(self.data)]
            loss = (weight * x - x * 0.8) ** 2
            weight -= lr * 2 * x * (weight * x - x * 0.8)
        self.history.append(f"trained lr={lr} → weight={weight:+.4f}")
        return loss

    def log(self) -> list[str]:
        return self.history


if __name__ == "__main__":
    with app.run():
        trainer = Trainer()
        print(trainer.normalize.remote())  # mutate live state before branching

        t0 = time.monotonic()
        branches = trainer.fork(3)
        print(f"forked 3 branches of the live trainer in {time.monotonic() - t0:.2f}s")

        lrs = [0.3, 0.01, 0.0005]
        futures = [b.train.spawn(lr) for b, lr in zip(branches, lrs)]
        for lr, future in zip(lrs, futures):
            print(f"  lr={lr:<6} → final loss {future.result():.6f}")

        # Each branch inherited the parent's history, then diverged.
        print("branch 0 remembers:", " | ".join(branches[0].log.remote()))

        # The parent is frozen now — it became the branch template.
        try:
            trainer.normalize.remote()
        except fleetlet.FrozenActorError as exc:
            print(f"parent is frozen, as expected: {type(exc).__name__}")

        # More branches later replay the frozen moment (no train entries).
        late = trainer.fork()
        print("late branch remembers:", " | ".join(late.log.remote()))
