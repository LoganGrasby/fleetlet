"""A real embedding service: BAAI/bge-small-en-v1.5 (384-dim, ONNX) behind
the HTTP gateway, with the model baked into the image at build time.

Serve it (locally or with cloud workers), or deploy it as a persistent
cloud service:

    python examples/07_embeddings.py                     # serve on :8283 ($PORT overrides)
    FLEETLET_TARGET=cloud python examples/07_embeddings.py
    fleetlet deploy examples/07_embeddings.py --name embed \\
        --net --cpus 2 --memory 1536

    curl -s -X POST localhost:8283/call/embed \\
         -d '{"texts": ["the cat sat on the mat", "GPUs go brrr"]}'

Throughput bench — grow one pool 1 → 2 → 4 workers and measure texts/sec
(each worker is its own microVM; on the cloud target, its own cloud VM):

    FLEETLET_TARGET=cloud python examples/07_embeddings.py --bench
"""

import os
import sys
import time

import fleetlet

MODEL = "BAAI/bge-small-en-v1.5"
CACHE = "/opt/models"

image = (
    fleetlet.Image.from_registry("python:3.12-slim")
    .pip_install("fastembed")
    .run_commands(
        f"mkdir -p {CACHE}",
        f"python3 -c 'from fastembed import TextEmbedding; "
        f"TextEmbedding(\"{MODEL}\", cache_dir=\"{CACHE}\")'",
    )
)

app = fleetlet.App("embeds", image=image)


@app.function(workers=1, cpus=2, memory=1536, net=True, timeout=300)
def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with bge-small-en-v1.5 (384 dims each)."""
    global _model
    if "_model" not in globals():
        from fastembed import TextEmbedding

        cache = CACHE if os.path.isdir(CACHE) else None
        _model = TextEmbedding(MODEL, cache_dir=cache)
    return [vec.tolist() for vec in _model.embed(texts)]


# ---------------------------------------------------------------- bench

def corpus(n: int) -> list[str]:
    themes = ["distributed systems", "sourdough baking", "orbital mechanics",
              "jazz harmony", "soil chemistry", "type theory", "whale migration"]
    return [f"note {i}: an observation about {themes[i % len(themes)]}, "
            f"item number {i} in the stream" for i in range(n)]


def bench(batches: int = 24, batch_size: int = 32) -> None:
    texts = corpus(batches * batch_size)
    chunks = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    print(f"bench: {len(texts)} texts as {batches} batches of {batch_size} "
          f"({app.target} target)\n")

    results = []
    with app.run():
        for workers in (1, 2, 4):
            t0 = time.monotonic()
            # map(workers=k) grows the pool; existing warm workers stay.
            list(embed.map(chunks[:workers], workers=workers))  # warm every worker
            grow_s = time.monotonic() - t0

            t0 = time.monotonic()
            out = list(embed.map(chunks, workers=workers))
            wall = time.monotonic() - t0

            assert len(out) == batches and len(out[0][0]) == 384
            tps = len(texts) / wall
            results.append((workers, wall, tps))
            print(f"  workers={workers}: {wall:5.1f}s for {len(texts)} texts "
                  f"→ {tps:6.1f} texts/s   (grow+warm took {grow_s:.1f}s)")

    base = results[0][2]
    print("\n  scaling vs 1 worker:")
    for workers, _, tps in results:
        print(f"    {workers} worker(s): {tps / base:4.2f}x")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        app.serve(port=int(os.environ.get("PORT", "8283")))
