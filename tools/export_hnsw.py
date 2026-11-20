#!/usr/bin/env python3
"""Build a Faiss HNSW index and export its graph to the project's formats.

Faiss is used only here. The search engine reads the exported files and never
links against Faiss.
"""

import argparse
import json
import os

import faiss
import numpy as np

INVALID = 0xFFFFFFFF


def read_vecs(path, dtype):
    """Reads .fvecs / .ivecs: each row is an int32 dimension then the values."""
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    if raw.size % (d + 1) != 0:
        raise ValueError(f"{path}: size is not a multiple of dimension {d} + 1")
    rows = raw.reshape(-1, d + 1)
    if not (rows[:, 0] == d).all():
        raise ValueError(f"{path}: rows disagree on dimension")
    return np.ascontiguousarray(rows[:, 1:]).view(dtype)


def write_matrix(path, x, dtype):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(x, dtype=dtype).tofile(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True, help=".fvecs database vectors")
    ap.add_argument("--query", required=True, help=".fvecs query vectors")
    ap.add_argument("--gt", help=".ivecs ground truth (not used with --limit)")
    ap.add_argument("--learn", help=".fvecs learn vectors (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="use only the first LIMIT base vectors")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, nargs="*", default=[16, 32, 64, 128, 256])
    args = ap.parse_args()
    if not args.limit and not args.gt:
        ap.error("--gt is required unless --limit is given")

    base = read_vecs(args.base, np.float32)
    queries = read_vecs(args.query, np.float32)
    if args.limit:
        base = base[: args.limit]
        # The shipped ground truth covers the full set, so recompute it.
        flat = faiss.IndexFlatL2(base.shape[1])
        flat.add(base)
        gt = flat.search(queries, 100)[1].astype(np.uint32)
    else:
        gt = read_vecs(args.gt, np.int32).astype(np.uint32)
    n, d = base.shape

    index = faiss.IndexHNSWFlat(d, args.M)
    index.hnsw.efConstruction = args.ef_construction
    index.add(base)
    hnsw = index.hnsw

    neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int64)
    offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    levels = faiss.vector_to_array(hnsw.levels)  # number of levels per node
    cum = [hnsw.cum_nb_neighbors(l) for l in range(hnsw.max_level + 2)]

    def level_lists(nodes, level):
        cols = np.arange(cum[level], cum[level + 1])
        lists = neighbors[offsets[nodes][:, None] + cols[None, :]]
        return np.where(lists < 0, INVALID, lists).astype(np.uint32)

    graph0 = level_lists(np.arange(n), 0)

    # Exactness check of the export against Faiss's own accessor.
    begin, end = np.zeros(1, np.uint64), np.zeros(1, np.uint64)
    for i in np.random.default_rng(0).integers(0, n, 1000).tolist():
        hnsw.neighbor_range(i, 0, faiss.swig_ptr(begin), faiss.swig_ptr(end))
        want = neighbors[int(begin[0]) : int(end[0])]
        want = np.where(want < 0, INVALID, want).astype(np.uint32)
        if not np.array_equal(graph0[i], want):
            raise RuntimeError(f"level-0 export mismatch at node {i}")
    valid = graph0[graph0 != INVALID]
    if valid.size and valid.max() >= n:
        raise RuntimeError("exported neighbor id out of range")

    os.makedirs(args.out, exist_ok=True)
    path = lambda name: os.path.join(args.out, name)
    write_matrix(path("base.fbin"), base, np.float32)
    write_matrix(path("query.fbin"), queries, np.float32)
    write_matrix(path("gt.ibin"), gt, np.uint32)
    write_matrix(path("graph0.bin"), graph0, np.uint32)
    if args.learn:
        write_matrix(path("learn.fbin"), read_vecs(args.learn, np.float32), np.float32)

    # upper.bin: entry point, max level, then per level 1..max_level:
    # node count, degree, sorted node ids, neighbor lists.
    with open(path("upper.bin"), "wb") as f:
        np.array([hnsw.entry_point, hnsw.max_level], dtype=np.uint32).tofile(f)
        for level in range(1, hnsw.max_level + 1):
            nodes = np.flatnonzero(levels > level)
            lists = level_lists(nodes, level)
            np.array(lists.shape, dtype=np.uint32).tofile(f)
            nodes.astype(np.uint32).tofile(f)
            lists.tofile(f)

    # Faiss's own search as a loose reference for recall and work per query.
    reference = {"faiss": faiss.__version__, "n": n, "d": d, "M": args.M,
                 "ef_construction": args.ef_construction, "k": args.k,
                 "entry_point": int(hnsw.entry_point), "max_level": int(hnsw.max_level),
                 "runs": []}
    for ef in args.ef:
        hnsw.efSearch = ef
        faiss.cvar.hnsw_stats.reset()
        found = index.search(queries, args.k)[1]
        recall = np.mean([len(set(a) & set(b)) for a, b in zip(found.tolist(), gt[:, : args.k].tolist())]) / args.k
        evals = faiss.cvar.hnsw_stats.ndis / len(queries)
        reference["runs"].append({"ef": ef, "recall": float(recall), "evals_per_query": float(evals)})
        print(f"faiss ef={ef}: recall@{args.k}={recall:.4f}, evals/query={evals:.1f}")
    with open(path("faiss_reference.json"), "w") as f:
        json.dump(reference, f, indent=2)
    print(f"wrote {n} x {d}, degree {graph0.shape[1]}, max level {hnsw.max_level} to {args.out}")


if __name__ == "__main__":
    main()
