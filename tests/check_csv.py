#!/usr/bin/env python3
"""Runs hvs_search on a synthetic set into a fresh CSV and checks every row
against the reference files written by tools/make_synthetic.py.

Hashes are recomputed here from their definitions in README.md, independently
of the C++ code. With --rejects it instead checks that bad command lines fail
before loading any data and write no CSV.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time

import numpy as np

COLUMNS = (
    "build_id tag host platform build compiler build_type cxx_flags "
    "logical_cpus omp_proc_bind "
    "omp_places dataset_hash n d degree queries k ef mode threads threads_used "
    "launch repetition warmups recall qps latency_mean_us latency_p50_us "
    "latency_p95_us latency_p99_us evals_per_query hops_per_query "
    "batches_per_query result_hash backend stage_timing verified gpu "
    "cuda_runtime cuda_driver wait_policy max_batch pinned resident init_s "
    "device_bytes launch_floor_us round_trips_per_query "
    "candidates_per_round_trip bytes_to_device_per_query "
    "bytes_from_device_per_query backend_us outside_backend_us gather_us "
    "submit_us wait_us to_device_us kernel_us from_device_us start_time_utc "
    "command"
).split()
# Which rows fill which columns from backend to from_device_us (README.md);
# verified is always filled.
HYBRID = ("backend stage_timing init_s round_trips_per_query "
          "candidates_per_round_trip").split()
CUDA = ("gpu cuda_runtime cuda_driver wait_policy max_batch pinned resident "
        "device_bytes launch_floor_us bytes_to_device_per_query "
        "bytes_from_device_per_query backend_us outside_backend_us "
        "gather_us").split()
THROUGHPUT = "submit_us wait_us".split()  # cuda, no --stage-timing
STAGED = "to_device_us kernel_us from_device_us".split()  # cuda, --stage-timing
EF = [10, 32, 64]  # the reference files the generator writes
LAUNCH = 7
TAG = "e2e"
# qps must be above this share of threads / mean latency, which is the share
# of the threads' wall time spent in queries. Alone it is 0.97 or more at 1
# thread, but under ctest -j other tests' threads can take the CPU between
# queries (down to 0.32 seen at 1 thread and 0.016 at 8, where starting the
# threads is a large part of a short pass), so the bound is loose at 1 thread
# and absent at 8.
QPS_SHARE = {1: 0.05, 8: 0.0}
STEP_NS = 10.0  # see the repetition check at the end of main


def fnv1a(data, h=0xCBF29CE484222325):
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def read_matrix(path):
    raw = np.fromfile(path, dtype=np.uint32)
    return raw[2:].reshape(int(raw[0]), int(raw[1]))


def result_hash(ids, work):
    """FNV-1a over little-endian uint32: query count, then per query its id,
    result count, ids, distance evaluations and hops."""
    values = [len(ids)]
    for q in range(len(ids)):
        values += [q, len(ids[q])] + ids[q].tolist() + work[q].tolist()
    return "%016x" % fnv1a(b"".join(int(v).to_bytes(4, "little") for v in values))


def fail(message):
    print("check_csv: " + message)
    sys.exit(1)


def check_rejects(args):
    """Each bad command line exits non-zero before loading data, no CSV."""
    out = os.path.join(args.scratch, "out.csv")
    bad = [["--mode", "cpu", "--verify"],
           ["--mode", "hybrid-sync", "--backend", "cpu", "--stage-timing"]]
    if args.no_cuda:
        bad.append(["--mode", "hybrid-sync", "--backend", "cuda"])
    for options in bad:
        if os.path.exists(out):
            os.remove(out)
        # The data directory does not exist, so loading it would fail too.
        command = [args.binary, "--data", os.path.join(args.scratch, "none"),
                   "--csv", out] + options
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode == 0 or os.path.exists(out):
            fail("%s did not fail without writing a CSV" % " ".join(options))
        if "base.fbin" in done.stderr:
            fail("%s loaded data before failing" % " ".join(options))
    # A CSV path that cannot be opened is refused before loading data too.
    missing = os.path.join(args.scratch, "missing", "out.csv")
    done = subprocess.run([args.binary, "--data", args.scratch, "--csv",
                           missing], capture_output=True, text=True)
    if done.returncode == 0 or "cannot open for appending" not in done.stderr:
        fail("an unwritable --csv path was not refused at startup")
    print("check_csv: %d command lines refused" % (len(bad) + 1))


def csv_safe(text):
    """The text as hvs_search writes it into one CSV field."""
    text = text.replace(",", ";")
    for bad in "\"'\n\r":
        text = text.replace(bad, "_")
    return text


def near(a, b, tolerance):
    return abs(a - b) <= tolerance + 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--binary", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--mode", default="cpu")
    ap.add_argument("--backend", default="")
    ap.add_argument("--stage-timing", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--query-count", type=int, default=0)  # 0: every query
    ap.add_argument("--rejects", action="store_true")
    ap.add_argument("--no-cuda", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.scratch, exist_ok=True)
    if args.rejects:
        return check_rejects(args)

    gt = read_matrix(os.path.join(args.data, "gt.ibin"))
    k = read_matrix(os.path.join(args.data, "ref_ef10.ibin")).shape[1]
    d = int(np.fromfile(os.path.join(args.data, "base.fbin"), np.uint32, 2)[1])
    out = os.path.join(args.scratch, "out.csv")
    if os.path.exists(out):
        os.remove(out)
    options = ["--mode", args.mode]
    if args.backend:
        options += ["--backend", args.backend]
    if args.stage_timing:
        options.append("--stage-timing")
    if args.verify:
        options.append("--verify")
    if args.query_count:
        options += ["--query-count", str(args.query_count)]
    # Three timed passes per ef, for the repetition check at the end, except
    # on the slower cuda backend, where that check does not apply.
    repetitions = 2 if args.backend == "cuda" else 3
    command = [args.binary, "--data", args.data, "--k", str(k),
               "--ef", ",".join(map(str, EF)), "--threads", str(args.threads),
               "--repetitions", str(repetitions), "--launch", str(LAUNCH),
               "--tag", TAG, "--csv", out] + options
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    build_id = subprocess.run([args.binary, "--build-id"], check=True,
                              capture_output=True, text=True).stdout.strip()

    with open(out, newline="") as f:
        lines = list(csv.reader(f))
    if lines[0] != COLUMNS:
        fail("header is %s" % lines[0])
    rows = [dict(zip(COLUMNS, line)) for line in lines[1:]]
    if any(len(line) != len(COLUMNS) for line in lines[1:]):
        fail("a row has the wrong number of fields")
    if len(rows) != len(EF) * repetitions:
        fail("%d rows, expected %d" % (len(rows), len(EF) * repetitions))

    dataset = 0xCBF29CE484222325
    for name in ("base.fbin", "query.fbin", "graph0.bin", "upper.bin",
                 "gt.ibin"):
        with open(os.path.join(args.data, name), "rb") as f:
            dataset = fnv1a(f.read(), dataset)

    hybrid = args.mode == "hybrid-sync"
    cuda = args.backend == "cuda"
    filled = {"verified"}
    if hybrid:
        filled |= set(HYBRID)
    if cuda:
        filled |= set(CUDA) | set(STAGED if args.stage_timing else THROUGHPUT)

    qps_by_ef = {}  # the qps strings of each ef's repetitions
    expected_keys = [(ef, r) for ef in EF for r in range(1, repetitions + 1)]
    for row, (ef, repetition) in zip(rows, expected_keys):
        at = "ef %d repetition %d: " % (ef, repetition)
        stem = os.path.join(args.data, "ref_ef%d" % ef)
        ids, work = read_matrix(stem + ".ibin"), read_matrix(stem + "_work.ibin")
        if args.query_count:  # only the first queries were searched
            ids, work = ids[:args.query_count], work[:args.query_count]
        n = len(ids)
        evals = int(work[:, 0].sum())
        hits = sum(int(np.isin(ids[q], gt[q, :k]).sum()) for q in range(n))
        expected = {
            "dataset_hash": "%016x" % dataset, "queries": str(n), "k": str(k),
            "ef": str(ef), "mode": args.mode, "threads": str(args.threads),
            "threads_used": str(args.threads), "launch": str(LAUNCH),
            "repetition": str(repetition), "warmups": "1", "tag": TAG,
            "recall": "%.6f" % (hits / (n * k)),
            "evals_per_query": "%.2f" % (evals / n),
            "hops_per_query": "%.2f" % (work[:, 1].sum() / n),
            "result_hash": result_hash(ids, work),
            "verified": "yes" if args.verify else "no",
        }
        if hybrid:
            expected.update({
                "backend": args.backend,
                "stage_timing": "1" if args.stage_timing else "0",
                "round_trips_per_query": row["batches_per_query"]})
        if cuda:
            expected.update({
                "wait_policy": "auto", "max_batch": row["degree"],
                "pinned": "1", "resident": "0",
                "bytes_to_device_per_query": "%.2f" % (evals * (4 + 4 * d) / n),
                "bytes_from_device_per_query": "%.2f" % (evals * 4 / n)})
        for name, value in expected.items():
            if row[name] != value:
                fail(at + "%s is %s, expected %s" % (name, row[name], value))
        for name in COLUMNS[COLUMNS.index("backend"):
                            COLUMNS.index("from_device_us") + 1]:
            if (row[name] != "") != (name in filled):
                fail(at + "%s is '%s', expected it %s" % (
                    name, row[name], "filled" if name in filled else "empty"))
        if len(row["build_id"]) != 12 or set(row["build_id"]) - set("0123456789abcdef"):
            fail("build_id %s is not 12 hex digits" % row["build_id"])
        if row["build_id"] != build_id:
            fail("build_id %s, but --build-id prints %s" % (row["build_id"],
                                                             build_id))
        flags = row["cxx_flags"].split()
        if not row["build_type"] or "-mavx2" not in flags:
            fail("build_type %s or cxx_flags %s not set" % (row["build_type"],
                                                            row["cxx_flags"]))
        if ("-DNDEBUG" in flags) != (row["build"] == "release"):
            fail("build %s does not match cxx_flags" % row["build"])
        if not (re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ",
                             row["start_time_utc"])
                and started <= row["start_time_utc"] <= finished):
            fail("start_time_utc %s is not between %s and %s" % (
                row["start_time_utc"], started, finished))
        if row["command"] != csv_safe(" ".join(command)):
            fail("command is %s" % row["command"])
        p50, p95, p99 = (float(row["latency_p%d_us" % p]) for p in (50, 95, 99))
        if not 0 < p50 <= p95 <= p99:
            fail("latency percentiles %s %s %s out of order" % (p50, p95, p99))
        # Every batch but the entry point's is one expansion (hop).
        if not 1 <= float(row["batches_per_query"]) <= float(
                row["hops_per_query"]) + 1.01:
            fail(at + "batches_per_query is outside 1 to hops + 1")

        # The latencies of a pass lie inside its wall time on each thread, so
        # threads / mean latency bounds qps from above; printed values are
        # rounded, hence the 0.005 and 0.05. Below it qps must stay above
        # QPS_SHARE of the bound (see QPS_SHARE).
        qps, mean = float(row["qps"]), float(row["latency_mean_us"])
        bound = args.threads * 1e6 / mean
        if not (bound * QPS_SHARE[args.threads] < qps
                <= args.threads * 1e6 / (mean - 0.005) + 0.05):
            fail(at + "qps %s is not within %.2f of threads / mean latency, "
                 "%.1f" % (row["qps"], QPS_SHARE[args.threads], bound))
        qps_by_ef.setdefault(ef, set()).add(row["qps"])

        # Sums of values printed with two digits, each off by up to 0.005.
        value = {name: float(row[name]) for name in filled - {"verified"}
                 if name not in ("backend", "gpu", "cuda_runtime",
                                 "cuda_driver", "wait_policy")}
        if any(v < 0 for v in value.values()):
            fail(at + "a backend column is negative")
        if hybrid:
            trips = value["round_trips_per_query"]
            per_trip = value["candidates_per_round_trip"]
            if not near(trips * per_trip, float(row["evals_per_query"]),
                        0.005 * (trips + per_trip) + 0.006):
                fail(at + "candidates per round trip times round trips is "
                     "not the evaluations")
        if cuda:
            if not (value["device_bytes"] > 0 and value["launch_floor_us"] > 0
                    and row["gpu"] and row["cuda_runtime"]):
                fail(at + "device columns are not set")
            # gather plus the three stages, or plus submit and wait, are
            # 75% to 100% of backend_us; the rest is copying out the results.
            split = STAGED if args.stage_timing else THROUGHPUT
            if not all(value[name] > 0 for name in split):
                fail(at + "a stage time is zero")
            stages = value["gather_us"] + sum(value[name] for name in split)
            backend = value["backend_us"]
            if not 0.75 * backend - 0.025 <= stages <= backend + 0.025:
                fail(at + "stage times add up to %.2f of backend_us %.2f" % (
                    stages, backend))
            # An independent bound: the time inside the backends is part of
            # the threads' wall time too.
            if qps > args.threads * 1e6 / (backend - 0.005) + 0.05:
                fail(at + "qps is above threads / backend_us")
            if not near(value["backend_us"] + value["outside_backend_us"],
                        float(row["latency_mean_us"]), 0.015):
                fail(at + "backend_us + outside_backend_us is not the mean "
                     "latency")
    # One step of the printed qps (0.1) is a wall time step of 0.1 n / qps^2
    # seconds. Where that is under STEP_NS, three passes print the same qps
    # only if their wall times agree to within STEP_NS, which passes that
    # vary by microseconds do not; so three equal values mean one qps was
    # reported for every repetition. Elsewhere equal values are not checked.
    for ef, values in qps_by_ef.items():
        step_ns = max(0.1 * n / float(v) ** 2 * 1e9 for v in values)
        if repetitions > 1 and len(values) == 1 and step_ns < STEP_NS:
            fail("ef %d: every repetition has qps %s" % (ef, values.pop()))
    print("check_csv: %d rows match the reference files" % len(rows))


if __name__ == "__main__":
    main()
