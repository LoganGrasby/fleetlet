"""`python3 -m fleetlet` — same as the `fleetlet` console script. This is
how deployed gateways launch inside cloud VMs, where only the staged package
(PYTHONPATH) exists and no entry-point shim is installed."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
