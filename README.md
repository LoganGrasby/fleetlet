# fleetlet

**Modal/Ray-style functions and actors on [smolvm](https://github.com/smol-machines/smolvm) microVMs.**
Define workers in code; run them in hardware-isolated VMs on your own machine — with fork-based
warm pools that clone a fully-loaded worker in ~100ms.

```python
import fleetlet

app = fleetlet.App("demo")

@app.function(workers=4)
def square(x: int) -> int:
    return x * x

if __name__ == "__main__":
    with app.run():
        print(square.remote(7))            # one call, one microVM
        print(list(square.map(range(10)))) # fan out across the fleet
```

Every worker is a real virtual machine (libkrun/KVM/Hypervisor.framework), not a container:
untrusted code gets a hypervisor boundary, its own kernel, CoW disks, and no network unless
you grant it. And because the engine can fork a *live* VM, a running actor can be
[branched mid-life](#branching-live-actors-instancefork) into divergent copies in ~0.1s.

## Why fork pools

Serverless platforms fight cold starts with memory snapshots. smolvm can do something
stronger: **fork a live VM**. fleetlet builds its scaling on that:

```
cold pool (pool="cold")                fork pool (pool="fork")
──────────────────────                 ───────────────────────
worker 0: boot+bake+import  ~10-60s    golden:  boot+bake+import+@enter  once
worker 1: boot+bake+import  ~10-60s    worker 0: fork golden             ~0.1-1s
worker 2: boot+bake+import  ~10-60s    worker 1: fork golden             ~0.1-1s
...                                    worker N: fork golden             ~0.1-1s
```

The golden VM boots, installs your image steps, imports your code, constructs your actor and
runs its `@fleetlet.enter` hooks (load the model, warm the cache…). It is then frozen, and every
worker is a copy-on-write clone that resumes with all of that state already in memory — the
runner is mid-`accept()` when it wakes up. Crashed workers are replaced by re-forking.

`workers > 1` uses a fork pool automatically; `pool="cold"` / `pool="fork"` overrides.

## Install

```bash
# 1. smolvm (the VM engine)
curl -sSL https://smolmachines.com/install.sh | bash

# 2. fleetlet
pip install fleetlet         # (from this repo: pip install -e .)

fleetlet doctor              # sanity check
```

## Actors (`@app.cls`)

Ray-style actors with Modal-style lifecycle hooks. One instance lives per worker; for fork
pools the constructor + `@enter` run **once**, in the golden:

```python
@app.cls(workers=4, pool="fork", image=fleetlet.Image.default().pip_install("torch"), net=True)
class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

    @fleetlet.enter
    def load(self):
        self.model = load_model(self.model_name)   # paid once, inherited by every clone

    def embed(self, text: str) -> list[float]:
        return self.model.encode(text)

with app.run():
    emb = Embedder("all-MiniLM-L6-v2")
    vectors = list(emb.embed.map(documents))
```

## Branching live actors (`instance.fork()`)

Fork pools clone a worker *template* built at boot. `fork()` goes further: it branches a
**running** actor — interpreter, heap, threads, everything — mid-life, in ~0.1s per branch:

```python
@app.cls(forkable=True)
class Session:
    @fleetlet.enter
    def boot(self):
        self.df = load_huge_dataset()      # expensive, happens once

    def prep(self): ...                    # mutate live state
    def train(self, lr: float): ...

with app.run():
    sess = Session()
    sess.prep.remote()                     # state evolves in the VM
    branches = sess.fork(3)                # 3 divergent copies of this exact moment
    losses = [b.train.remote(lr) for b, lr in zip(branches, [0.3, 0.01, 0.0005])]
```

Every branch starts from the same in-memory moment and then diverges — best-of-N
exploration, speculative "try it in a copy of reality" execution, per-test fixture
isolation — without ever rebuilding the expensive state.

Semantics (they mirror the engine exactly):

- **Forking freezes the parent.** smolvm supports one live branch point per VM: the first
  fork snapshots the parent as an immutable template. Method calls on it then raise
  `FrozenActorError`, while further `fork()` calls mint more branches *at the frozen state*
  (want the mainline to continue? take an extra branch as the new mainline).
- **Branches are full actors** — `.remote/.spawn/.map`, own VM, own divergent state — but
  they can't `fork()` again (a clone can't be re-forked), and a dead branch is not
  replaced: its state exists nowhere else.
- `fork()` serializes with in-flight calls through the actor's queue, so the snapshot never
  catches a call mid-flight. Local target only for now.

## Images

Modal-style immutable chained builders. Steps are baked inside the guest right after first
boot (once per cold worker; once total for a fork pool):

```python
image = (
    fleetlet.Image.from_registry("python:3.12-slim")
    .pip_install("pandas", "pyarrow")
    .apt_install("ffmpeg")
    .run_commands("mkdir -p /data")
    .env(TZ="UTC")
)

@app.function(image=image, net=True)   # build steps need egress → net=True
def crunch(path: str): ...
```

`Image.default()` is `python:X.Y-slim` matching your host interpreter, which keeps pickled
lambdas/closures bytecode-compatible.

### Packed images (offline / air-gapped / rate-limit-proof)

`machine start` pulls from the registry inside the VM. If that's not viable — no egress,
corporate DNS, Docker Hub anonymous rate limits — pack the image **on the host** once and
boot workers from the artifact (sub-second starts, no pull ever):

```bash
smolvm pack create --image python:3.14-slim --no-sign --output py314.smolmachine
# writes py314.smolmachine (launcher) + py314.smolmachine.smolmachine (the artifact)
```

```python
app = fleetlet.App("etl", image=fleetlet.Image.from_smolmachine("py314.smolmachine.smolmachine"))
```

`FLEETLET_DEFAULT_IMAGE` overrides `Image.default()` everywhere without touching code —
set it to a registry tag or a `.smolmachine` path (useful in CI and on air-gapped hosts).

## Calling conventions

| call | behavior |
|---|---|
| `f.remote(*a, **kw)` | run in a worker VM, block, return the value |
| `f.spawn(*a, **kw)` | returns a `concurrent.futures.Future` |
| `f.map(xs, ys, workers=8)` | builtin-`map` semantics, ordered results, scales the pool |
| `f.local(*a, **kw)` | run in-process, no VM |

Exceptions raised in the guest are re-raised on the host (with the remote traceback attached);
`map(..., return_exceptions=True)` yields them instead. `retries=N` re-runs calls that die from
*worker failure* (never user exceptions). Remote `print()` output is relayed to your terminal.

## Function options

```python
@app.function(
    image=image,          # default: app.image or Image.default()
    workers=4,            # pool size (map can grow it)
    pool="auto",          # auto | fork | cold
    cpus=2, memory=1024,  # per-worker vCPUs / MiB
    net=False,            # True, or an allow-list: net=["api.example.com"]
    env={"X": "1"},
    volumes={"/host/data": "/data"},   # cold pools; untested with fork
    timeout=60.0,         # per-call seconds
    retries=1,            # transport-level retries
    gpu=False,            # Vulkan (virtio-gpu)
    cuda=False,           # host NVIDIA GPU over vsock (see smolvm docs)
    share_weights=False,  # fork pools + CUDA: one VRAM copy of frozen weights
    forkable=False,       # actors only: enable instance.fork() (local target)
)
```

With `cuda=True, share_weights=True`, sibling clones share a single VRAM copy of the golden's
frozen model weights (fleetlet sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for you).

## How code reaches the VM

- Your project directory (cwd by default, `App(project_root=...)` to override) is tarred and
  synced into every worker at `/opt/fleetlet/project`; module-level functions ship as
  `module:qualname` references and are imported from that copy — Python-version-safe.
- Functions defined in your entrypoint script are handled like Modal handles `__main__`:
  the guest loads the file and aliases it, so your script-level classes unpickle fine.
- Lambdas and closures ship by value via cloudpickle (fleetlet copies your host cloudpickle
  into the guest — no pip needed). These carry bytecode: keep guest and host Python minors equal.
- The in-VM runner is a single stdlib-only file speaking length-prefixed pickle frames over a
  forwarded TCP port. One connection per worker, one call at a time per worker.

## HTTP serving

Every `@app.function` doubles as an HTTP endpoint — FastMCP's bargain, applied to VMs:
declaring the function was the whole job. Routes, parameter schemas, request validation,
and machine-readable docs are generated from the signature; isolation, pooling, and
retries come from the decorator options. Any language can call in.

```python
@app.function(workers=2)
def embed(text: str, dims: int = 8) -> list[float]:
    """Deterministic toy embedding."""
    ...

if __name__ == "__main__":
    app.serve(port=8283)        # or: fleetlet serve script.py
```

```bash
curl -s localhost:8283/ | jq                                   # index: schemas + docs
curl -s -X POST localhost:8283/call/embed -d '{"text": "hi"}'  # {"ok": true, "result": [...]}
curl -s -X POST localhost:8283/call/embed -d '{"dims": "big"}' # 400 + which field and why
```

- `POST /call/<fn>` waits and returns; `POST /spawn/<fn>` returns a `call_id` to poll at
  `GET /result/<id>` (handed out once; unfetched results expire after ~15 min).
  `GET /` lists everything; `GET /health` for probes.
- Bodies are JSON parameter objects validated against the generated schema (typed errors,
  `400` before any VM is touched). Type hints → schema: primitives, containers,
  `Optional`, `Literal`; unhinted params accept anything.
- Results must be JSON-serializable; Python clients can send `Accept: application/x-pickle`
  for arbitrary objects. Pickle is never accepted as *input*.
- `fleetlet serve script.py --local` runs functions in-process (dev loop, no VMs);
  `--warm` boots all pools before accepting traffic.
- v0.1: no auth, binds `127.0.0.1` — front it with a real proxy to share it.

Since schemas travel with the functions, exposing them as **MCP tools** (VM-isolated tool
execution for agents) is a thin adapter away with FastMCP:

```python
from fastmcp import FastMCP

mcp = FastMCP("fleet")
with app.run():
    for slug, fn in app._functions.items():
        mcp.tool(fn.remote, name=slug, description=fn.__doc__ or "")
    mcp.run()  # every tool call executes inside a microVM
```

## Cloud

The same code runs against the **smol cloud** ([smol-machines/smol](https://github.com/smol-machines/smol)) —
managed microVMs instead of your machine. Set your tenant key and flip the target:

```bash
export SMOL_CLOUD_TOKEN=smk_...        # same variable the smol SDK/CLI use

fleetlet run script.py --cloud        # or: FLEETLET_TARGET=cloud python script.py
```

```python
app = fleetlet.App("etl", target="cloud")   # or leave unset → $FLEETLET_TARGET
```

`.remote()`, `.map()`, actors, crash-replacement, and `fleetlet serve --cloud`
(local gateway, cloud workers) all behave identically. Under the hood each worker is a
cloud machine created over the REST API; the host talks to its in-VM runner through the
machine's HTTPS ingress URL, which the platform already gates behind your tenant key
(plus a per-run token fleetlet adds on top). Measured from a laptop: worker up in
~11–15s (parallel for pools), then ~0.2–0.5s per call round-trip.

### Deploying a service

`fleetlet deploy` runs your HTTP gateway **persistently in a cloud microVM** with a
stable HTTPS URL — the Modal-deploy analog:

```bash
fleetlet deploy examples/05_http_service.py --name svc
# → https://flt-deploy-svc-<tenant>.apps.smolmachines.com

curl -H "Authorization: Bearer $SMOL_CLOUD_TOKEN" \
     -X POST https://flt-deploy-svc-<tenant>.apps.smolmachines.com/call/embed \
     -d '{"text": "hi"}'
```

The project is shipped into the VM and served with `--local` semantics (calls run in
the service VM's own process — one hardware-isolated microVM per service). The URL is
**tenant-private**: unauthenticated requests get a 401 at the platform edge, so the
gateway's own no-auth v0.1 posture is safe out of the box. Deploying again under the
same name replaces the service. Flags: `--port`, `--cpus`, `--memory`, `--net`
(egress for the VM), `--app` (pick an App variable).

### Cloud specifics

- **Fork pools work on the cloud too.** The golden VM stages code, bakes the image,
  loads your state, then clones inherit its warm RAM — the golden's running runner
  included (same process, same in-memory state). Clones have no
  ingress URL, so their calls ride the cloud **exec API** straight to the inherited
  runner on localhost: fully API-authenticated, zero open ports. Two trade-offs:
  per-call overhead is one exec round-trip, and single results are capped at ~14MB
  (the exec stdout limit). If a golden lands on a node without a fork-capable
  engine, affected workers transparently provision cold instead. Today they're at
  their best with pure-Python warm state; for native-heavy model loading, use the
  default cold pools (as `07_embeddings.py` does).
- **Leak safety net:** worker machines carry a 1h server-side TTL — if your process
  dies without teardown, the cloud deletes them anyway. `FLEETLET_CLOUD_TTL` tunes it
  (`0` disables). Deployments never get a TTL.
- `gpu`/`cuda`/`volumes`/`share_weights` and packed `.smolmachine` images are
  local-only (clear ConfigError on the cloud target). Images come from the
  platform's registry mirror.
- `fleetlet ls --cloud` / `clean --cloud` manage cloud machines; `clean` spares
  `flt-deploy-*` services unless you pass `--deployments`.

## CLI

```bash
fleetlet run script.py [--cloud]     # run a script (same as python script.py)
fleetlet serve script.py [--cloud]   # serve over HTTP (--port/--warm/--local)
fleetlet deploy script.py            # persistent HTTPS service on the smol cloud
fleetlet ls [--cloud]                # list fleet machines (flt-*)
fleetlet clean [--app X] [--cloud [--deployments]]   # delete leaked workers
fleetlet doctor                      # verify smolvm / python / cloud auth
```

`with app.run():` tears down every VM it started (also via atexit). Machines are named
`flt-{app}-{function}-{digest}-{run_id}-w{n}` (`-g` for goldens), so cleanup is always possible.

## Limitations (v0.1)

- Cloud fork-clone calls go over the exec API (no per-clone ingress yet platform-side):
  ~an exec round-trip per call, results ≤ ~14MB. Deployed gateways run functions
  in-process (`--local`) rather than fanning out to their own worker VMs.
- Images need `python3` inside (use `python:*-slim` bases or bake it).
- One in-flight call per worker; concurrency = pool size.
- Arguments/results must be picklable and travel through host RAM.
- Image bake steps run in-guest and need `net=True`; no layer cache yet
  (`pack --from-vm` snapshot caching is the planned fix).
- CUDA/`share_weights` options are plumbed but only exercised on Linux hosts with the
  smolvm CUDA daemon; `--gpu` (Vulkan) is untested from fleetlet.
- `instance.fork()` is local-only, and one branch point deep: forking freezes the parent,
  and branches can't re-fork (engine constraints — cloud actor forks are on the roadmap).

## Examples

```bash
python examples/01_hello.py      # your first VM function
python examples/02_map.py        # fork pool fan-out
python examples/03_actor.py      # actor with @enter, model loaded once
python examples/04_isolation.py  # hostile code stays contained, workers auto-replace
python examples/05_http_service.py  # functions as HTTP endpoints (schema'd + validated)
python examples/06_cloud.py         # same code on smol cloud machines (+ deploy)
python examples/07_embeddings.py    # real model (bge-small ONNX) as a service; --bench
python examples/08_branching.py     # fork a LIVE actor: one warm state, divergent futures
python examples/09_database.py      # seed postgres once, fork the running DB per test
python examples/10_speculative.py   # try 3 risky migrations in forked realities, keep 1
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
