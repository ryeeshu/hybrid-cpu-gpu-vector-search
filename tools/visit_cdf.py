#!/usr/bin/env python3
"""Turns visit-count files into the curves of visit_cdf.csv.

Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
Each curve gives, for the most-visited share of base vectors, the share of
all distance evaluations they received. Curves: in-sample (vectors ranked by
their own counts), held-out (ranked by the first half of the queries, counted
on the second half) and uniform (the same evaluations spread at random).
Before writing, it checks that every curve rises to 1, that held-out lies
below in-sample, in-sample above uniform and uniform above y = x, and that
the in-sample shares agree with the percentages hvs_search printed.
"""

import argparse
import csv
import math
import os
import re
import sys

import numpy as np

# The runs sweep.sh makes: name -> (ef, query range as "whole", "first" or
# "second" half of the learn set).
RUNS = {
    "all_ef16": (16, "whole"),
    "all_ef64": (64, "whole"),
    "all_ef256": (256, "whole"),
    "first_ef64": (64, "first"),
    "second_ef64": (64, "second"),
}
FIXED_SHARES = [0.01, 0.05, 0.10, 0.25, 0.50]
POINTS = 200
# Curves are compared at this precision. hvs_search prints percentages with
# one decimal, so a printed share is within half a decimal of the curve.
TOLERANCE = 1e-9
PRINTED_TOLERANCE = 0.0005 + TOLERANCE
COLUMNS = ("build_id dataset_hash ef k curve query_range queries "
           "total_evaluations never_visited entry_point share_of_vectors "
           "share_of_evaluations").split()
# Keys hvs_search prints, first; the later spellings are accepted only until
# a built binary has been checked.
ALIASES = {
    "total_evaluations": ("evaluations", "total_evaluations", "total_evals"),
    "never_visited": ("never_visited", "never_visited_count"),
    "entry_point": ("entry_point", "entry"),
    "range": ("range", "visit_range", "query_range"),
}


def fail(message):
    print("visit_cdf: " + message, file=sys.stderr)
    sys.exit(1)


def read_counts(path):
    """Counts of a visit file: uint32 n, uint32 1, then n uint32 counts."""
    raw = np.fromfile(path, dtype="<u4")
    if len(raw) < 2 or raw[1] != 1 or len(raw) != 2 + int(raw[0]):
        fail("%s is not a visit-count file" % path)
    return raw[2:].astype(np.int64)


def read_text(path):
    """key=value lines of a run's output, keys in lower case with '_'. Keys
    hold no digits, so the summary line ("... at ef=64: ...") is skipped."""
    values = {}
    with open(path) as f:
        for line in f:
            match = re.match(r"\s*([A-Za-z][A-Za-z_ -]*?)\s*=\s*(.*?)\s*$",
                             line)
            if match:
                key = re.sub(r"[ -]", "_", match.group(1).lower())
                values[key] = match.group(2)
    return values


def read_printed_shares(path):
    """Percent of evaluations hvs_search printed for the most-visited 1, 5,
    10, 25 and 50 percent of vectors, as {percent of vectors: percent}; empty
    if the file has no such lines."""
    printed = {}
    pattern = (r"\s*most-visited\s+(\d+)% of vectors:\s+([0-9.]+)% of "
               r"evaluations")
    with open(path) as f:
        for line in f:
            match = re.match(pattern, line)
            if match:
                printed[int(match.group(1))] = float(match.group(2))
    return printed


def lookup(values, name, path, required=True):
    for key in ALIASES.get(name, (name,)):
        if key in values:
            return values[key]
    if required:
        fail("%s has no %s=" % (path, name))
    return None


def expected_range(which, learn_rows):
    half = learn_rows // 2
    return {"whole": (0, learn_rows), "first": (0, half),
            "second": (half, half)}[which]


def load_run(name, counts_dir, text_dir, learn_rows):
    """One run: its counts and the facts printed with them."""
    ef, which = RUNS[name]
    text_path = os.path.join(text_dir, name + ".txt")
    counts = read_counts(os.path.join(counts_dir, name + ".bin"))
    values = read_text(text_path)
    run = {"counts": counts, "ef": ef,
           "range": expected_range(which, learn_rows),
           "total": int(counts.sum()), "never": int((counts == 0).sum()),
           "printed": read_printed_shares(text_path)}
    if int(lookup(values, "ef", text_path)) != ef:
        fail("%s was run at ef %s, not %d" % (text_path, values["ef"], ef))
    for key in ("build_id", "dataset_hash", "k", "entry_point"):
        run[key] = lookup(values, key, text_path)
    # Facts the program printed must agree with the file and the plan.
    for key, mine in (("total_evaluations", run["total"]),
                      ("never_visited", run["never"])):
        printed = lookup(values, key, text_path, required=False)
        if printed is not None and int(printed) != mine:
            fail("%s says %s=%s, the counts give %d" % (text_path, key,
                                                       printed, mine))
    printed = lookup(values, "range", text_path, required=False)
    if printed is not None:
        numbers = [int(x) for x in re.findall(r"\d+", printed)]
        if len(numbers) == 2 and tuple(numbers) != run["range"]:
            fail("%s says range=%s, the sweep asked for %d,%d" % (
                text_path, printed, run["range"][0], run["range"][1]))
    return run


def share_grid(n):
    """Vector counts j: about POINTS log-spaced from 1 to n, plus the fixed
    shares, so share_of_vectors = j / n is exact."""
    logs = np.round(np.logspace(0, math.log10(n), POINTS))
    fixed = [round(s * n) for s in FIXED_SHARES]
    grid = np.unique(np.concatenate([logs, fixed, [1, n]]).astype(np.int64))
    return grid[(grid >= 1) & (grid <= n)]


def ranked_curve(order_key, counts, grid, rng):
    """Share of evaluations in `counts` held by the top j vectors by
    order_key; ties in order_key are broken by a seeded shuffle."""
    tie = rng.permutation(len(counts))
    order = np.lexsort((tie, -order_key))
    running = np.cumsum(counts[order])
    return running[grid - 1] / running[-1]


def poisson_curve(mean, shares):
    """Expected in-sample curve when counts are Poisson(mean): vectors ranked
    by count, the evaluations they hold as a share of all, interpolated
    linearly inside a run of equal counts. Computed analytically from the
    probability mass function (no random numbers), for a large vector set."""
    sd = math.sqrt(mean)
    low = max(0, int(mean - 12 * sd - 20))
    high = int(mean + 12 * sd + 20)
    value = np.arange(low, high + 1)
    log_fact = np.array([math.lgamma(v + 1) for v in value])
    p = np.exp(value * math.log(mean) - mean - log_fact)
    p /= p.sum()
    value, p = value[::-1], p[::-1]  # largest counts first
    vectors = np.cumsum(p)
    evals = np.cumsum(p * value) / (p * value).sum()
    idx = np.minimum(np.searchsorted(vectors, shares), len(p) - 1)
    before_v = np.where(idx > 0, vectors[idx - 1], 0.0)
    before_e = np.where(idx > 0, evals[idx - 1], 0.0)
    result = before_e + (shares - before_v) * value[idx] / (
        (p * value).sum())
    return np.minimum(result, 1.0)


def check_curves(curves, shares, held_out_limit):
    """Stops on a curve that cannot be right. curves is a list of (run,
    curve name, values). Every curve must not fall and must end at 1. The
    held-out curve must lie below the in-sample curve of the same counts,
    the in-sample curves above the uniform ones, and those above y = x."""
    by_key = {}
    for run, curve, values in curves:
        label = "%s curve at ef %d" % (curve, run["ef"])
        if np.any(np.diff(values) < -TOLERANCE):
            fail("the %s falls somewhere" % label)
        if abs(values[-1] - 1.0) > TOLERANCE:
            fail("the %s ends at %g, not 1" % (label, values[-1]))
        by_key[(curve, run["ef"])] = values
    if np.any(by_key[("held-out", 64)] > held_out_limit + TOLERANCE):
        fail("the held-out curve is above the in-sample curve of the same "
             "counts")
    for (curve, ef), values in by_key.items():
        if curve == "in-sample" and np.any(
                values < by_key[("uniform", ef)] - TOLERANCE):
            fail("the in-sample curve at ef %d is below the uniform one" % ef)
        if curve == "uniform" and np.any(values < shares - TOLERANCE):
            fail("the uniform curve at ef %d is below y = x" % ef)


def check_printed(runs, in_sample, grid, n):
    """Compares the in-sample curve of each run with the percentages
    hvs_search printed for it; returns how many it compared. in_sample maps
    a run name to its curve over grid."""
    compared = 0
    for name, curve in in_sample.items():
        for share in FIXED_SHARES:
            percent = round(100 * share)
            printed = runs[name]["printed"].get(percent)
            if printed is None:
                continue
            vectors = int(share * n)  # hvs_search counts whole vectors
            at = np.searchsorted(grid, vectors)
            if vectors == 0:
                mine = 0.0
            elif at < len(grid) and grid[at] == vectors:
                mine = curve[at]
            else:
                continue
            compared += 1
            if abs(printed / 100.0 - mine) > PRINTED_TOLERANCE:
                fail("%s prints %.1f%% of evaluations for the top %d%% of "
                     "vectors, the counts give %.1f%%" % (
                         name + ".txt", printed, percent, 100 * mine))
    return compared


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--counts-dir", required=True,
                    help="directory of <run>.bin visit-count files")
    ap.add_argument("--text-dir", required=True,
                    help="directory of <run>.txt key=value outputs")
    ap.add_argument("--learn-rows", type=int, required=True,
                    help="rows of learn.fbin")
    ap.add_argument("--out", required=True, help="visit_cdf.csv to write")
    args = ap.parse_args()

    runs = {name: load_run(name, args.counts_dir, args.text_dir,
                           args.learn_rows) for name in RUNS}
    n = len(runs["all_ef64"]["counts"])
    for key in ("build_id", "dataset_hash"):
        if len({run[key] for run in runs.values()}) != 1:
            fail("the runs disagree on %s" % key)
    if any(len(run["counts"]) != n for run in runs.values()):
        fail("the runs disagree on the number of vectors")
    if min(run["total"] for run in runs.values()) == 0:
        fail("a run has no distance evaluations")

    grid = share_grid(n)
    shares = grid / n
    rng = np.random.default_rng(0)  # tie-breaking for the held-out ranking
    curves = []  # (run, curve name, share of evaluations at each grid point)
    in_sample = {}
    for name in ("all_ef16", "all_ef64", "all_ef256"):
        run = runs[name]
        counts = run["counts"]
        in_sample[name] = ranked_curve(counts, counts, grid, rng)
        curves.append((run, "in-sample", in_sample[name]))
        curves.append((run, "uniform",
                       poisson_curve(run["total"] / n, shares)))
    # Ranked by the first half's counts, evaluations from the second half.
    first, second = runs["first_ef64"], runs["second_ef64"]
    held_out = ranked_curve(first["counts"], second["counts"], grid, rng)
    curves.append((second, "held-out", held_out))

    # The halves ranked by their own counts, for the checks only.
    in_sample["first_ef64"] = ranked_curve(first["counts"], first["counts"],
                                           grid, rng)
    in_sample["second_ef64"] = ranked_curve(second["counts"],
                                            second["counts"], grid, rng)
    check_curves(curves, shares, in_sample["second_ef64"])
    compared = check_printed(runs, in_sample, grid, n)

    rows = []
    for run, curve, curve_shares in curves:
        start, count = run["range"]
        for share_v, share_e in zip(shares, curve_shares):
            rows.append([run["build_id"], run["dataset_hash"], run["ef"],
                         run["k"], curve, "%d:%d" % (start, start + count),
                         count, run["total"], run["never"],
                         run["entry_point"], "%.10g" % share_v,
                         "%.10g" % share_e])

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    print("visit_cdf: %d rows, %d vectors, written to %s; %d printed "
          "percentages agree" % (len(rows), n, args.out, compared))


if __name__ == "__main__":
    main()
