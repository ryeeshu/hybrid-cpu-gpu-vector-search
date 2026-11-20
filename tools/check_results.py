#!/usr/bin/env python3
"""Checks the search.csv and verify.csv of a sweep against expected hashes.

Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
Exit status 1 with one line per problem, 0 with a one-line summary.
"""

import argparse
import csv
import os
import sys

EF_VALUES = [16, 32, 64, 128, 256]
THREADS = [1, 8]
# Timed configurations: name, mode, backend, stage_timing, threads. The
# backend and stage_timing columns are empty in cpu mode.
CONFIGS = [
    ("v1_t1", "cpu", "", "", 1),
    ("v1_t8", "cpu", "", "", 8),
    ("v2_t1", "hybrid-sync", "cuda", "0", 1),
    ("v2_t8", "hybrid-sync", "cuda", "0", 8),
    ("v2stage_t1", "hybrid-sync", "cuda", "1", 1),
    ("v2stage_t8", "hybrid-sync", "cuda", "1", 8),
    ("v2cpu_t1", "hybrid-sync", "cpu", "0", 1),
    ("v2cpu_t8", "hybrid-sync", "cpu", "0", 8),
]
REQUIRED = (
    "build_id dataset_hash queries k ef mode backend stage_timing threads "
    "threads_used launch repetition verified result_hash recall "
    "evals_per_query hops_per_query batches_per_query build"
).split()
# Expected columns and the decimals they are compared at.
COMPARED = [("recall", 6), ("evals_per_query", 2), ("hops_per_query", 2),
            ("batches_per_query", 2)]
# Columns that must hold one value over all rows, when the CSV has them.
SINGLE = ["build_id", "dataset_hash", "build", "build_type", "cxx_flags",
          "host"]
MAX_LINES = 50

problems = []


def problem(text):
    problems.append(text)


def decimals(value, places):
    """The value as a number printed with that many decimals, or as is."""
    try:
        return "%.*f" % (places, float(value))
    except ValueError:
        return value


def read_expected(path):
    table = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["dataset_hash"], int(row["queries"]), int(row["k"]),
                   int(row["ef"]))
            table[key] = row
    return table


def read_rows(path):
    """Rows of a CSV as dicts with a 'line' number, or None if unusable."""
    name = os.path.basename(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            problem("%s: missing columns %s" % (name, " ".join(missing)))
            return None
        rows = []
        for row in reader:
            row["line"] = reader.line_num
            rows.append(row)
    if not rows:
        problem("%s: no rows" % name)
    return rows


def as_int(row, column, name):
    try:
        return int(row[column])
    except (TypeError, ValueError):
        problem("%s line %d: %s '%s' is not a number" % (
            name, row["line"], column, row[column]))
        return None


def row_key(row, name):
    """(mode, backend, stage_timing, threads, queries, ef, launch, repetition)"""
    numbers = [as_int(row, c, name)
               for c in ("threads", "queries", "ef", "launch", "repetition")]
    if None in numbers:
        return None
    threads, queries, ef, launch, repetition = numbers
    return (row["mode"], row["backend"], row["stage_timing"], threads,
            queries, ef, launch, repetition)


def check_against_table(rows, name, table):
    for row in rows:
        at = "%s line %d: " % (name, row["line"])
        try:
            key = (row["dataset_hash"], int(row["queries"]), int(row["k"]),
                   int(row["ef"]))
        except ValueError:
            problem(at + "queries, k or ef is not a number")
            continue
        expected = table.get(key)
        if expected is None:
            problem(at + "dataset_hash %s, queries %d, k %d, ef %d is not "
                    "in the expected table" % key)
            continue
        where = " (queries %d, ef %d, %s threads %s)" % (
            key[1], key[3], row["mode"] + "/" + row["backend"],
            row["threads"])
        if row["result_hash"] != expected["result_hash"]:
            problem(at + "result_hash %s, expected %s%s" % (
                row["result_hash"], expected["result_hash"], where))
        for column, places in COMPARED:
            if decimals(row[column], places) != decimals(expected[column],
                                                         places):
                problem(at + "%s %s, expected %s%s" % (
                    column, row[column], expected[column], where))
        if row["threads_used"] != row["threads"]:
            problem(at + "threads_used %s is not threads %s" % (
                row["threads_used"], row["threads"]))


def check_unique(rows, name):
    seen = {}
    for row in rows:
        key = row_key(row, name)
        if key is None:
            continue
        if key in seen:
            problem("%s line %d: duplicate of line %d (mode %s, backend '%s', "
                    "stage_timing '%s', threads %d, queries %d, ef %d, "
                    "launch %d, repetition %d)" % ((name, row["line"],
                                                    seen[key]) + key))
        seen[key] = row["line"]
    return seen


def check_single(all_rows):
    for column in SINGLE:
        values = {}
        for name, rows in all_rows:
            for row in rows:
                if column in row:
                    values.setdefault(row[column], name)
        if len(values) > 1:
            problem("%s has %d values: %s" % (column, len(values),
                                              ", ".join(sorted(values))))
    for name, rows in all_rows:
        if any(row["build"] != "release" for row in rows):
            problem("%s: a row is not from a release build" % name)


def check_verify(rows, name, full):
    covered = set()
    for row in rows:
        at = "%s line %d: " % (name, row["line"])
        if row["verified"] != "yes":
            problem(at + "verified is '%s', not yes" % row["verified"])
        if row["mode"] != "hybrid-sync" or row["backend"] != "cuda":
            problem(at + "mode %s, backend '%s', not hybrid-sync with cuda" %
                    (row["mode"], row["backend"]))
        if full is not None and row["queries"] != str(full):
            problem(at + "queries %s, not the full set of %d" %
                    (row["queries"], full))
        covered.add((row["threads"], row["ef"]))
    for threads in THREADS:
        for ef in EF_VALUES:
            if (str(threads), str(ef)) not in covered:
                problem("%s: no row for threads %d, ef %d" % (name, threads,
                                                              ef))


def parse_extra(texts):
    """{launch: set of names} from LAUNCH:NAME[,NAME...] options; exits on
    text of another form or on a repeated launch."""
    extra = {}
    for text in texts:
        launch, _, names = text.partition(":")
        if not launch.isdigit() or not names:
            sys.exit("check_results: --extra needs LAUNCH:NAME[,NAME...], "
                     "not '%s'" % text)
        if int(launch) in extra:
            sys.exit("check_results: --extra launch %s is given twice" %
                     launch)
        extra[int(launch)] = set(names.split(","))
    return extra


def expected_search_keys(rounds, repetitions, queries, full, extra):
    """Key -> description for every row search.csv must have. Extra maps a
    launch number above the rounds to the names that have rows for it."""
    wanted = {}
    runs = [(name, mode, backend, stage, threads, queries)
            for name, mode, backend, stage, threads in CONFIGS]
    if queries != full:
        runs += [("v1full_t%d" % t, "cpu", "", "", t, full) for t in THREADS]
    names = {run[0] for run in runs}
    for launch, named in sorted(extra.items()):
        if launch <= rounds:
            problem("--extra launch %d is not above the %d rounds" % (
                launch, rounds))
        for name in sorted(named - names):
            problem("--extra launch %d names %s, which is not a "
                    "configuration of this sweep" % (launch, name))
    for name, mode, backend, stage, threads, q in runs:
        launches = list(range(1, rounds + 1)) + [
            launch for launch, named in sorted(extra.items())
            if name in named and launch > rounds]
        for launch in launches:
            for ef in EF_VALUES:
                for repetition in range(1, repetitions + 1):
                    key = (mode, backend, stage, threads, q, ef, launch,
                           repetition)
                    wanted[key] = (name, launch, ef, repetition)
    return wanted


def check_complete(seen, wanted, name):
    # One line per configuration and round.
    absent = {}
    for key, (config, launch, ef, repetition) in wanted.items():
        if key not in seen:
            absent.setdefault((config, launch, key[4]), []).append(
                "%d/%d" % (ef, repetition))
    for (config, launch, queries), where in absent.items():
        problem("%s: %s round %d (queries %d) is missing %d rows, as "
                "ef/repetition: %s" % (name, config, launch, queries,
                                       len(where), " ".join(where)))
    for key, line in seen.items():
        if key not in wanted:
            problem("%s line %d: unexpected row (mode %s, backend '%s', "
                    "stage_timing '%s', threads %d, queries %d, ef %d, "
                    "launch %d, repetition %d)" % ((name, line) + key))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("results", help="results directory of a sweep")
    ap.add_argument("--expected",
                    default=os.path.join(here, "expected_hashes.csv"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--query-count", type=int,
                    help="--query-count of the timed launches, if used")
    ap.add_argument("--full-queries", type=int,
                    help="queries in query.fbin (default: those of verify.csv)")
    ap.add_argument("--build-id", help="id every row must carry")
    ap.add_argument("--extra", action="append", default=[],
                    metavar="LAUNCH:NAME[,NAME...]",
                    help="rows of one extra launch, after the rounds, for "
                    "these configurations only; may be repeated")
    args = ap.parse_args()
    extra = parse_extra(args.extra)

    table = read_expected(args.expected)
    loaded = {}
    all_rows = []
    for name in ("search.csv", "verify.csv"):
        path = os.path.join(args.results, name)
        loaded[name] = read_rows(path) if os.path.exists(path) else None
        if loaded[name]:
            all_rows.append((name, loaded[name]))
    if not all_rows and not problems:
        problem("%s has neither search.csv nor verify.csv" % args.results)
    search = loaded["search.csv"]
    verify = loaded["verify.csv"]
    full = args.full_queries
    if full is None and verify:
        full = as_int(verify[0], "queries", "verify.csv")
    if full is None and search:
        counts = [int(r["queries"]) for r in search if r["queries"].isdigit()]
        full = max(counts) if counts else None

    for name, rows in all_rows:
        check_against_table(rows, name, table)
        if args.build_id:
            for row in rows:
                if row["build_id"] != args.build_id:
                    problem("%s line %d: build_id %s, the sweep built %s" % (
                        name, row["line"], row["build_id"], args.build_id))
    check_single(all_rows)
    if verify:
        check_unique(verify, "verify.csv")
        check_verify(verify, "verify.csv", full)
    if search:
        seen = check_unique(search, "search.csv")
        for row in search:
            if row["verified"] != "no":
                problem("search.csv line %d: verified is '%s' on a timed "
                        "launch" % (row["line"], row["verified"]))
        queries = args.query_count if args.query_count else full
        if full is not None:
            check_complete(seen, expected_search_keys(
                args.rounds, args.repetitions, queries, full, extra),
                "search.csv")

    if problems:
        for text in problems[:MAX_LINES]:
            print("check_results: " + text, file=sys.stderr)
        if len(problems) > MAX_LINES:
            print("check_results: and %d more problems" %
                  (len(problems) - MAX_LINES), file=sys.stderr)
        sys.exit(1)
    counts = ", ".join("%d rows in %s" % (len(rows), name)
                       for name, rows in all_rows)
    ids = {row["build_id"] for _, rows in all_rows for row in rows}
    text = "check_results: ok: %s, build_id %s, %d rounds" % (
        counts, ids.pop(), args.rounds)
    for launch, names in sorted(extra.items()):
        text += ", extra launch %d for %s" % (launch, " ".join(sorted(names)))
    print(text)


if __name__ == "__main__":
    main()
