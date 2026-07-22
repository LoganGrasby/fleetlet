"""Speculative execution: try risky changes in forked realities, keep one.

An agent (or a nervous human) has three candidate migrations for a live
in-memory service. Instead of gambling the mainline on one of them, fork
the service and run EVERY candidate for real — actual code, actual
mutations, actual crashes — each in its own hardware-isolated copy of
reality. Health-check the outcomes, promote the branch that survived,
discard the rest. The mainline state never sees a single write.

The nastiest failure mode is on display: candidate B crashes MIDWAY,
leaving half-migrated state — the kind of partial write that is agony to
roll back in place, and costs nothing to throw away as a branch.

    python examples/10_speculative.py
"""

import textwrap

import fleetlet

app = fleetlet.App("specdemo")


@app.cls(forkable=True)
class UserService:
    @fleetlet.enter
    def boot(self):
        import random

        random.seed(11)
        self.users = {
            i: {"name": f"user{i}", "plan": random.choice(["free", "pro"]),
                "credits": random.randrange(100)}
            for i in range(20_000)
        }

    def apply_migration(self, code: str) -> dict:
        """Run an untrusted migration against the LIVE user table. It may
        be wrong, and it may die halfway through — in a branch, so what."""
        exec(code, {"users": self.users})
        return self.health()

    def health(self) -> dict:
        migrated = sum(1 for u in self.users.values() if "tier" in u)
        legacy = sum(1 for u in self.users.values() if "plan" in u)
        return {"users": len(self.users), "migrated": migrated, "legacy": legacy}


# Goal: rename every user's "plan" field to "tier". Three attempts:
MIGRATIONS = {
    "A: overzealous cleanup": """
        for uid in list(users):
            if users[uid]["plan"] == "free":
                del users[uid]              # "surely nobody needs free users"
            else:
                users[uid]["tier"] = users[uid].pop("plan")
    """,
    "B: crashes midway": """
        for i, uid in enumerate(list(users)):
            if i == 12_000:
                raise RuntimeError("hit a malformed record")   # partial write!
            users[uid]["tier"] = users[uid].pop("plan")
    """,
    "C: does it right": """
        for uid in users:
            users[uid]["tier"] = users[uid].pop("plan")
    """,
}


def acceptable(report: dict, baseline: dict) -> bool:
    return (report["users"] == baseline["users"]
            and report["migrated"] == baseline["users"]
            and report["legacy"] == 0)


if __name__ == "__main__":
    with app.run():
        service = UserService()
        baseline = service.health.remote()
        print(f"mainline: {baseline}\n")

        branches = service.fork(len(MIGRATIONS))
        winner = None
        for (name, code), branch in zip(MIGRATIONS.items(), branches):
            try:
                report = branch.apply_migration.remote(textwrap.dedent(code))
                verdict = "PROMOTE" if acceptable(report, baseline) else "discard"
            except RuntimeError as exc:
                report = branch.health.remote()   # inspect the wreckage
                verdict = f"discard (crashed: {exc})"
            print(f"  {name:24} → {report}  [{verdict}]")
            if verdict == "PROMOTE":
                winner = branch

        service = winner  # the surviving reality becomes the new mainline
        print(f"\npromoted C; mainline is now: {service.health.remote()}")
        print("the two corrupted realities are discarded with their VMs — "
              "no rollback code was ever written")
