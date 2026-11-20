#!/usr/bin/env python3
"""Deterministic synthetic dataset, graph and reference search results.

Needs only numpy, so tests do not depend on Faiss or on SIFT1M. Vectors are
integer-valued by default, so squared L2 is exact in float32. Exact ties are
rare at the default settings; a low-dimensional set such as
--d 6 --sigma 4 --clusters 8 makes them common and exercises the
(distance, id) tie-break.
"""

import argparse
import heapq
import os

import numpy as np

INVALID = 0xFFFFFFFF


def write_matrix(path, x, dtype):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        np.ascontiguousarray(x, dtype=dtype).tofile(f)


def make_points(rng, centers, n, sigma, as_float):
    x = centers[rng.integers(0, len(centers), n)] + rng.normal(0, sigma, (n, centers.shape[1]))
    x = np.clip(x, 0, 255)
    return (x if as_float else np.rint(x)).astype(np.float32)


def all_distances(a, b):
    """Squared L2 from every row of a to every row of b (exact for integers)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = (a64 * a64).sum(1)[:, None] + (b64 * b64).sum(1)[None, :] - 2.0 * a64 @ b64.T
    return np.maximum(d, 0.0)


def nearest(dist_row, count):
    """Ids of the count smallest entries, ordered by (distance, id)."""
    count = min(count, len(dist_row))
    kth = np.partition(dist_row, count - 1)[count - 1]
    part = np.flatnonzero(dist_row <= kth)  # every entry tied with the cut
    order = np.lexsort((part, dist_row[part]))
    return part[order][:count]


def build_graph(rng, base, knn, extra):
    """knn nearest neighbors plus extra random edges per node, so the graph
    stays connected. Self loops and duplicates become INVALID, which leaves
    holes anywhere in a row, not only at its end."""
    n = len(base)
    graph = np.full((n, knn + extra), INVALID, dtype=np.uint32)
    for start in range(0, n, 1000):
        dist = all_distances(base[start : start + 1000], base)
        for i in range(len(dist)):
            dist[i, start + i] = np.inf
            graph[start + i, :knn] = nearest(dist[i], knn)
    random_edges = rng.integers(0, n, (n, extra))
    for i in range(n):
        seen = set(graph[i, :knn].tolist()) | {i}
        for j, v in enumerate(random_edges[i].tolist()):
            if v not in seen:
                seen.add(v)
                graph[i, knn + j] = v
    return graph


def reference_search(base, graph, entry, q, k, ef):
    """Level-0 best-first search. Returns the top-k ids, the number of
    distance evaluations and the number of expansions (hops).

    Rules an implementation must follow to reproduce this id for id:
    - Every comparison is on (distance, id): both heaps, the stop test, the
      admission test and the final sort.
    - A node is marked visited when it is first discovered, not when popped.
    - evals = 1 for the entry point plus every newly visited neighbor. No
      neighbor is filtered out before its distance is computed.
    - INVALID slots are skipped, never treated as the end of a row.
    - hops counts popped candidates that passed the stop test, including
      those with no new neighbors.
    - Neighbor order within a row does not affect the result.
    - The admission threshold may be read once per expansion instead of once
      per neighbor, but only if a result set with fewer than ef entries counts
      as an infinite threshold.
    - With ef < k the result has ef entries.
    """

    def dist(ids):
        diff = base[ids] - q
        return (diff * diff).sum(axis=1, dtype=np.float32)

    d0 = float(dist([entry])[0])
    visited = {entry}
    candidates = [(d0, entry)]  # min-heap
    results = [(-d0, -entry)]  # max-heap on (distance, id)
    evals = 1
    hops = 0
    while candidates:
        d, c = heapq.heappop(candidates)
        worst = (-results[0][0], -results[0][1])
        if len(results) == ef and (d, c) > worst:
            break
        hops += 1
        new = []
        for v in graph[c].tolist():
            if v != INVALID and v not in visited:
                visited.add(v)
                new.append(v)
        if not new:
            continue
        evals += len(new)
        for v, dv in zip(new, dist(new).tolist()):
            worst = (-results[0][0], -results[0][1])
            if len(results) < ef or (dv, v) < worst:
                heapq.heappush(candidates, (dv, v))
                heapq.heappush(results, (-dv, -v))
                if len(results) > ef:
                    heapq.heappop(results)
    top = sorted((-d, -v) for d, v in results)[:k]
    return [v for _, v in top], evals, hops


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=32)
    ap.add_argument("--sigma", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--float", action="store_true", help="non-integer coordinates")
    ap.add_argument("--ef", type=int, nargs="*", default=[10, 32, 64])
    args = ap.parse_args()
    if args.n <= 16:
        ap.error("--n must exceed the 16 nearest neighbors each node links to")

    rng = np.random.default_rng(args.seed)
    centers = rng.integers(0, 256, (args.clusters, args.d)).astype(np.float64)
    base = make_points(rng, centers, args.n, args.sigma, args.float)
    queries = make_points(rng, centers, args.nq, args.sigma, args.float)

    dist = all_distances(queries, base)
    gt = np.stack([nearest(row, args.k) for row in dist]).astype(np.uint32)

    graph = build_graph(rng, base, knn=16, extra=16)
    to_mean = all_distances(base.mean(axis=0, keepdims=True), base)[0]
    entry = int(nearest(to_mean, 1)[0])

    os.makedirs(args.out, exist_ok=True)
    path = lambda name: os.path.join(args.out, name)
    write_matrix(path("base.fbin"), base, np.float32)
    write_matrix(path("query.fbin"), queries, np.float32)
    write_matrix(path("gt.ibin"), gt, np.uint32)
    write_matrix(path("graph0.bin"), graph, np.uint32)
    with open(path("upper.bin"), "wb") as f:
        np.array([entry, 0], dtype=np.uint32).tofile(f)  # entry point, max level

    for ef in args.ef:
        found = [reference_search(base, graph, entry, q, args.k, ef) for q in queries]
        ids = np.array([f[0] for f in found], dtype=np.uint32)
        work = np.array([f[1:] for f in found], dtype=np.uint32)  # evals, hops
        write_matrix(path(f"ref_ef{ef}.ibin"), ids, np.uint32)
        write_matrix(path(f"ref_ef{ef}_work.ibin"), work, np.uint32)
        recall = np.mean([len(set(a) & set(b)) for a, b in zip(ids.tolist(), gt.tolist())]) / args.k
        print(f"ef={ef}: recall@{args.k}={recall:.4f}, mean evals={work[:, 0].mean():.1f}, "
              f"mean hops={work[:, 1].mean():.1f}")
    print(f"wrote {args.n} x {args.d} to {args.out}, entry point {entry}")


if __name__ == "__main__":
    main()
