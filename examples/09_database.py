"""Fork a REAL, RUNNING database per test — fixture isolation by construction.

The eternal testing tradeoff: rebuild the database per test (slow) or share
it across tests (flaky — one leaked write poisons the suite). Transaction
rollback only covers what runs inside a transaction; it can't undo a
dropped table, a schema change, or a killed connection.

fork() dissolves the tradeoff. PostgreSQL is installed, started, and seeded
ONCE in a live actor; each "test" then gets a fork of that VM — inheriting
the *running* postmaster and all 500k rows in ~1s. Tests can DROP TABLE,
corrupt every row, or kill the server: each one is trashing its own
private reality, and the frozen template stays pristine for the next fork.

    python examples/09_database.py    (first boot bakes postgres: ~20-60s)
"""

import time

import fleetlet

app = fleetlet.App("dbdemo")

pg_image = fleetlet.Image.default().apt_install("postgresql")


@app.cls(forkable=True, image=pg_image, net=True, cpus=2, memory=1536)
class SeededDB:
    @fleetlet.enter
    def seed(self):
        import subprocess

        def sh(cmd: str) -> None:
            subprocess.run(["sh", "-c", cmd], check=True,
                           capture_output=True, text=True)

        # Minimal guests have no /etc/hosts; postgres wants "localhost".
        sh("grep -qs localhost /etc/hosts || echo '127.0.0.1 localhost' >> /etc/hosts")
        sh("pg_ctlcluster $(ls /etc/postgresql) main start")
        sh("su postgres -s /bin/sh -c 'createdb shop'")
        print("postgres up — seeding 500,000 orders…")
        self.sql("""
            create table orders(
                id serial primary key, sku text, qty int, total numeric(10,2));
            insert into orders(sku, qty, total)
                select 'sku-' || g, (g % 7) + 1, ((g % 997) * 1.37)::numeric(10,2)
                from generate_series(1, 500000) g;
        """)

    def sql(self, script: str) -> list[str]:
        """Run SQL against the live server; returns non-empty output lines."""
        import subprocess

        proc = subprocess.run(
            ["su", "postgres", "-s", "/bin/sh", "-c",
             "psql -d shop -tA -v ON_ERROR_STOP=1"],
            input=script, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip())
        return [line for line in proc.stdout.splitlines() if line.strip()]


# Each "test" trashes (or reads) what it believes is THE database.
TESTS = {
    "drops the whole table": ("""
        drop table orders;
        select count(*) from pg_tables where tablename = 'orders';
    """, "0"),
    "poisons every row": ("""
        update orders set total = -1;
        select count(*) from orders where total < 0;
    """, "500000"),
    "sees zero poisoned rows": ("""
        select count(*) from orders where total < 0;
    """, "0"),
}

if __name__ == "__main__":
    with app.run():
        db = SeededDB()

        t0 = time.monotonic()
        rows = db.sql.remote("select count(*) from orders")[0]
        print(f"fixture built ONCE in {time.monotonic() - t0:.0f}s "
              f"(bake postgres + seed {int(rows):,} rows)")

        t0 = time.monotonic()
        branches = db.fork(len(TESTS))
        print(f"forked {len(TESTS)} live database copies "
              f"in {time.monotonic() - t0:.2f}s\n")

        futures = [branch.sql.spawn(script)
                   for branch, (script, _) in zip(branches, TESTS.values())]
        for (name, (_, expected)), future in zip(TESTS.items(), futures):
            got = future.result()[-1]
            status = "PASS" if got == expected else f"FAIL (want {expected})"
            print(f"  test {name!r:26} → {got:>8}  {status}")

        # Every test mutated "the" database — and none of them saw another.
        late = db.fork()
        pristine = late.sql.remote("select count(*) from orders")[0]
        assert pristine == rows
        print(f"\nnext fork is pristine: {int(pristine):,} rows "
              "(the fixture was never rebuilt)")
