# plot.py: summary table, report tables and figures from saved benchmark CSVs.
# Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
#
# Usage: plot.py RESULTS_DIR [RESULTS_DIR ...] --out FIGURE_DIR
# Every number comes from the CSVs in the results directories. Input that
# contradicts itself is refused (exit 1, one line per problem).

import argparse
import os
import sys
import textwrap

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Configurations (mode, backend and stage_timing together).
V1 = "V1"
V2 = "V2"
V2_STAGE = "V2 stage-timing"
V2_CPU = "V2 CPU distances"

THREADS = (1, 8)
OPERATING_EF = 64
FLAG_PERCENT = 5.0
# A table cell gets a dagger when its range over all rows exceeds this share
# of the reported QPS; table (d) also lists unflagged configurations above
# ROW_LIST_PERCENT.
ROW_MARK_PERCENT = 10.0
ROW_LIST_PERCENT = 8.0
# V1 at 8 threads on fewer queries than this is a pass of a few milliseconds.
SHORT_PASS_QUERIES = 5000
# A V2 round trip must cost this many times more thread time at 8 threads
# than at 1 for a caption to say that threads queue behind each other.
QUEUE_RATIO = 1.5
DAGGER = "†"
MAX_PROBLEM_LINES = 50

KEY = ["mode", "backend", "stage_timing", "threads", "queries", "ef",
       "launch", "repetition"]
TEXT_COLUMNS = ["host", "platform", "build_id", "dataset_hash", "mode",
                "backend", "stage_timing", "result_hash", "gpu"]
NUMBER_COLUMNS = (
    "threads queries k ef launch repetition recall qps latency_mean_us "
    "latency_p50_us latency_p95_us latency_p99_us evals_per_query "
    "hops_per_query batches_per_query launch_floor_us round_trips_per_query "
    "candidates_per_round_trip bytes_to_device_per_query "
    "bytes_from_device_per_query backend_us outside_backend_us gather_us "
    "submit_us wait_us to_device_us kernel_us from_device_us").split()
# Numbers every row must have.
ROW_NUMBERS = ("threads queries k ef launch repetition recall qps "
               "latency_mean_us evals_per_query").split()
# Columns a row of each configuration must have filled.
NEEDED = {
    V1: [],
    V2: ["gpu", "launch_floor_us", "round_trips_per_query",
         "candidates_per_round_trip", "backend_us", "outside_backend_us",
         "gather_us", "submit_us", "wait_us"],
    V2_STAGE: ["gpu", "launch_floor_us", "round_trips_per_query",
               "candidates_per_round_trip", "backend_us",
               "outside_backend_us", "to_device_us", "kernel_us",
               "from_device_us"],
    V2_CPU: ["round_trips_per_query", "candidates_per_round_trip"],
}
# Columns copied from the one detail row of a summary line.
DETAIL_COLUMNS = (
    "recall evals_per_query round_trips_per_query candidates_per_round_trip "
    "latency_mean_us latency_p50_us latency_p95_us latency_p99_us backend_us "
    "outside_backend_us gather_us submit_us wait_us to_device_us kernel_us "
    "from_device_us launch_floor_us result_hash").split()
SUMMARY_COLUMNS = (
    "config threads queries k ef rows launches recall evals_per_query "
    "round_trips_per_query candidates_per_round_trip qps_reported "
    "qps_launch_min qps_launch_max qps_row_min qps_row_max spread_pct flag "
    "row_spread_pct qps_vs_v1 detail_launch detail_repetition latency_mean_us "
    "latency_p50_us latency_p95_us latency_p99_us backend_us "
    "outside_backend_us gather_us submit_us wait_us to_device_us kernel_us "
    "from_device_us launch_floor_us result_hash").split()
XFER_COLUMNS = ("build_id host platform gpu cpu_threads run pairs "
                "ship_pinned_us gather_us resident_us cpu_1t_us "
                "cpu_all_us").split()
XFER_NUMBERS = ("cpu_threads run pairs ship_pinned_us gather_us resident_us "
                "cpu_1t_us cpu_all_us").split()
CDF_COLUMNS = ("ef curve share_of_vectors share_of_evaluations "
               "dataset_hash build_id").split()
SHARES = [0.01, 0.05, 0.10, 0.25, 0.50]

# Colors: Okabe-Ito. One color per version; line style and marker per thread
# count; one color per stage, the same in every figure.
VERSION_COLOR = {V1: "#0072B2", V2: "#D55E00"}
VERSION_LABEL = {V1: "V1 (CPU only)", V2: "V2 (CPU + GPU)"}
THREAD_STYLE = {1: ("-", "o"), 8: ("--", "s")}
STAGE_COLOR = {
    "outside": "#999999", "other": "#D9D9D9", "gather": "#009E73",
    "submit": "#CC79A7", "wait": "#7A3B63", "to_device": "#56B4E9",
    "kernel": "#E69F00", "from_device": "#005A8D",
}
FIGSIZE = (6.5, 4.0)

problems = []


def problem(text):
    problems.append(text)


def refuse_if_problems():
    if not problems:
        return
    for text in problems[:MAX_PROBLEM_LINES]:
        print("plot: " + text, file=sys.stderr)
    if len(problems) > MAX_PROBLEM_LINES:
        print("plot: and %d more problems" %
              (len(problems) - MAX_PROBLEM_LINES), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------- reading

def line_list(frame):
    """The file line numbers of a frame, as short text."""
    numbers = [str(n) for n in frame["line"].tolist()]
    text = ", ".join(numbers[:5])
    if len(numbers) > 5:
        text += " and %d more" % (len(numbers) - 5)
    return "line " + text if len(numbers) == 1 else "lines " + text


def read_text_csv(path):
    """A CSV with every value as text and a 'line' column; None if unusable."""
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as error:
        problem("%s: cannot read: %s" % (path, error))
        return None
    if frame.empty:
        problem("%s: no rows" % path)
        return None
    frame["line"] = frame.index + 2
    return frame


def missing_columns(frame, wanted, path):
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        problem("%s: missing columns %s" % (path, " ".join(missing)))
    return bool(missing)


def to_numbers(frame, columns, path):
    """Columns as numbers; blank stays NaN, other text is a problem."""
    ok = True
    for column in columns:
        if column not in frame.columns:
            continue
        numbers = pd.to_numeric(frame[column], errors="coerce")
        bad = frame[numbers.isna() & (frame[column] != "")]
        if not bad.empty:
            problem("%s %s: %s '%s' is not a number" % (
                path, line_list(bad), column, bad[column].iloc[0]))
            ok = False
        frame[column] = numbers
    return ok


def config_of(mode, backend, stage):
    if mode == "cpu" and backend == "" and stage == "":
        return V1
    if mode == "hybrid-sync" and backend == "cuda" and stage == "0":
        return V2
    if mode == "hybrid-sync" and backend == "cuda" and stage == "1":
        return V2_STAGE
    if mode == "hybrid-sync" and backend == "cpu" and stage == "0":
        return V2_CPU
    return None


def load_search(directory):
    """search.csv of a directory with a 'config' column, or None."""
    path = os.path.join(directory, "search.csv")
    if not os.path.exists(path):
        problem("%s: no search.csv" % directory)
        return None
    frame = read_text_csv(path)
    if frame is None:
        return None
    wanted = TEXT_COLUMNS + NUMBER_COLUMNS
    if missing_columns(frame, wanted, path):
        return None
    if not to_numbers(frame, NUMBER_COLUMNS, path):
        return None
    for column in ROW_NUMBERS:
        bad = frame[frame[column].isna()]
        if not bad.empty:
            problem("%s %s: %s is empty" % (path, line_list(bad), column))
            return None
    for column in ("threads", "queries", "k", "ef", "launch", "repetition"):
        frame[column] = frame[column].astype(int)
    frame["config"] = [config_of(m, b, s) for m, b, s in zip(
        frame["mode"], frame["backend"], frame["stage_timing"])]
    unknown = frame[frame["config"].isna()]
    if not unknown.empty:
        problem("%s %s: mode '%s', backend '%s', stage_timing '%s' is not "
                "a known configuration" % (
                    path, line_list(unknown), unknown["mode"].iloc[0],
                    unknown["backend"].iloc[0],
                    unknown["stage_timing"].iloc[0]))
        return None
    check_search(frame, path)
    return frame


def check_search(frame, path):
    """Appends a problem for every way the rows contradict each other."""
    check_single_values(frame, path)
    check_unique_keys(frame, path)
    check_hashes(frame, path)
    check_needed(frame, path)
    check_sums(frame, path)
    if "threads_used" in frame.columns:
        bad = frame[frame["threads_used"] != frame["threads"].astype(str)]
        if not bad.empty:
            problem("%s %s: threads_used is not threads" % (
                path, line_list(bad)))


def check_single_values(frame, path):
    gpu_rows = frame[frame["gpu"] != ""]
    columns = [("host", frame), ("platform", frame), ("build_id", frame),
               ("dataset_hash", frame), ("k", frame), ("gpu", gpu_rows)]
    for column, rows in columns:
        values = sorted(rows[column].unique().tolist(), key=str)
        if len(values) > 1:
            problem("%s: %s has %d values: %s" % (
                path, column, len(values), ", ".join(map(str, values))))
    if "build" in frame.columns and (frame["build"] != "release").any():
        problem("%s: a row is not from a release build" % path)


def check_unique_keys(frame, path):
    dup = frame[frame.duplicated(subset=KEY, keep=False)]
    for key, rows in dup.groupby(KEY):
        problem("%s %s: same key (mode %s, backend '%s', stage_timing '%s', "
                "threads %d, queries %d, ef %d, launch %d, repetition %d)" %
                ((path, line_list(rows)) + key))


def check_hashes(frame, path):
    """One result_hash, recall and evals_per_query per (queries, k, ef)."""
    for (queries, k, ef), rows in frame.groupby(["queries", "k", "ef"]):
        where = "queries %d, k %d, ef %d" % (queries, k, ef)
        hashes = sorted(rows["result_hash"].unique())
        if len(hashes) > 1:
            problem("%s: %s has %d result_hash values: %s" % (
                path, where, len(hashes), " ".join(hashes)))
        elif rows["recall"].round(6).nunique() > 1:
            problem("%s: %s has one result_hash but several recall values" %
                    (path, where))
        elif rows["evals_per_query"].round(2).nunique() > 1:
            problem("%s: %s has one result_hash but several "
                    "evals_per_query values" % (path, where))


def check_needed(frame, path):
    for config, columns in NEEDED.items():
        rows = frame[frame["config"] == config]
        for column in columns:
            empty = rows[column].isna() | (rows[column] == "")
            if empty.any():
                problem("%s %s: %s row has no %s" % (
                    path, line_list(rows[empty]), config, column))


def check_sums(frame, path):
    """Backend plus outside time is the latency; parts fit in the backend."""
    for config in (V2, V2_STAGE):
        rows = frame[frame["config"] == config]
        total = rows["backend_us"] + rows["outside_backend_us"]
        off = (total - rows["latency_mean_us"]).abs() > (
            0.01 * rows["latency_mean_us"] + 0.1)
        if off.any():
            problem("%s %s: backend_us + outside_backend_us is not "
                    "latency_mean_us" % (path, line_list(rows[off])))
    rows = frame[frame["config"] == V2]
    parts = rows["gather_us"] + rows["submit_us"] + rows["wait_us"]
    over = parts > 1.01 * rows["backend_us"] + 0.1
    if over.any():
        problem("%s %s: gather_us + submit_us + wait_us exceeds backend_us" %
                (path, line_list(rows[over])))


def load_xfer(directory, info):
    path = os.path.join(directory, "xfer.csv")
    if not os.path.exists(path):
        print("%s: no xfer.csv, skipping the xfer figure" % directory)
        return None
    frame = read_text_csv(path)
    if frame is None or missing_columns(frame, XFER_COLUMNS, path):
        return None
    if not to_numbers(frame, XFER_NUMBERS, path):
        return None
    if frame[XFER_NUMBERS].isna().any().any():
        problem("%s: an empty number in %s" % (path, " ".join(
            c for c in XFER_NUMBERS if frame[c].isna().any())))
        return None
    for column in ("host", "platform", "build_id", "gpu", "cpu_threads"):
        values = sorted(frame[column].astype(str).unique())
        if len(values) > 1:
            problem("%s: %s has %d values: %s" % (
                path, column, len(values), ", ".join(values)))
    for column in ("host", "platform", "build_id"):
        if frame[column].iloc[0] != info[column]:
            problem("%s: %s is %s, search.csv has %s" % (
                path, column, frame[column].iloc[0], info[column]))
    if info["gpu"] and frame["gpu"].iloc[0] != info["gpu"]:
        problem("%s: gpu is %s, search.csv has %s" % (
            path, frame["gpu"].iloc[0], info["gpu"]))
    dup = frame[frame.duplicated(subset=["run", "pairs"], keep=False)]
    if not dup.empty:
        problem("%s %s: same run and pairs twice" % (path, line_list(dup)))
    return frame


def load_cdf(directory, info):
    path = os.path.join(directory, "visit_cdf.csv")
    if not os.path.exists(path):
        print("%s: no visit_cdf.csv, skipping the visits figure" % directory)
        return None
    frame = read_text_csv(path)
    if frame is None or missing_columns(frame, CDF_COLUMNS, path):
        return None
    if not to_numbers(frame, ["ef", "queries", "share_of_vectors",
                              "share_of_evaluations"], path):
        return None
    for column in ("build_id", "dataset_hash"):
        values = sorted(frame[column].unique())
        if values != [info[column]]:
            problem("%s: %s is %s, search.csv has %s" % (
                path, column, " ".join(values), info[column]))
    return frame


def dataset_name(search):
    """SIFT1M for 1,000,000 vectors of 128 dimensions, else the sizes."""
    sizes = sorted(set(zip(search["n"], search["d"])))
    if sizes == [("1000000", "128")]:
        return "SIFT1M"
    return "dataset of " + ", ".join("%s vectors of %s dimensions" % size
                                     for size in sizes)


def load_directory(directory):
    """Everything plot.py reads from one results directory."""
    search = load_search(directory)
    if search is None:
        return None
    info = {
        "dir": directory,
        "host": search["host"].iloc[0],
        "platform": search["platform"].iloc[0],
        "build_id": search["build_id"].iloc[0],
        "dataset_hash": search["dataset_hash"].iloc[0],
        "k": int(search["k"].iloc[0]),
        "gpu": next((g for g in search["gpu"] if g), ""),
        "dataset": dataset_name(search),
        "max_queries": int(search["queries"].max()),
    }
    if any(sep in info["host"] for sep in ("/", "\\")) or info["host"] in (
            "", ".", ".."):
        problem("%s: host '%s' cannot be a directory name" % (
            directory, info["host"]))
    return {"info": info, "search": search,
            "xfer": load_xfer(directory, info),
            "cdf": load_cdf(directory, info)}


# ---------------------------------------------------------------- summary

def median_row(rows):
    """The row with the median qps; the lower one when the count is even."""
    ordered = rows.sort_values(["qps", "repetition"])
    return ordered.iloc[(len(ordered) - 1) // 2]


def summarize_group(key, group):
    """One summary line for one (config, threads, queries, ef)."""
    config, threads, queries, ef = key
    launches = {n: rows for n, rows in group.groupby("launch")}
    launch_qps = {n: rows["qps"].median() for n, rows in launches.items()}
    ordered = sorted(launch_qps, key=lambda n: (launch_qps[n], n))
    middle = ordered[(len(ordered) - 1) // 2]
    detail = median_row(launches[middle])
    line = {"config": config, "threads": threads, "queries": queries,
            "k": detail["k"], "ef": ef, "rows": len(group),
            "launches": len(launches), "detail_launch": middle,
            "detail_repetition": detail["repetition"]}
    for column in DETAIL_COLUMNS:
        line[column] = detail[column]
    if config == V2_STAGE:
        # Stage-timing runs are not a throughput measurement.
        return line
    values = list(launch_qps.values())
    reported = float(np.median(values))
    line.update({"qps_reported": reported, "qps_launch_min": min(values),
                 "qps_launch_max": max(values),
                 "qps_row_min": group["qps"].min(),
                 "qps_row_max": group["qps"].max()})
    if len(values) >= 2:
        line["spread_pct"] = 100.0 * (max(values) - min(values)) / reported
        line["flag"] = int(line["spread_pct"] > FLAG_PERCENT)
    line["row_spread_pct"] = 100.0 * (line["qps_row_max"]
                                      - line["qps_row_min"]) / reported
    return line


def summarize(search):
    keys = ["config", "threads", "queries", "ef"]
    lines = [summarize_group(key, group)
             for key, group in search.groupby(keys)]
    summary = pd.DataFrame(lines)
    summary = summary.reindex(columns=SUMMARY_COLUMNS)
    for column in ("threads", "queries", "k", "ef", "rows", "launches",
                   "detail_launch", "detail_repetition"):
        summary[column] = summary[column].astype(int)
    summary["qps_vs_v1"] = ratios_to_v1(summary)
    order = {V1: 0, V2: 1, V2_STAGE: 2, V2_CPU: 3}
    summary["order"] = summary["config"].map(order)
    summary = summary.sort_values(["queries", "order", "threads", "ef"])
    return summary.drop(columns="order").reset_index(drop=True)


def ratios_to_v1(summary):
    """qps_reported over V1's qps_reported at the same threads, queries, ef."""
    v1 = summary[summary["config"] == V1]
    base = {(r.threads, r.queries, r.ef): r.qps_reported
            for r in v1.itertuples()}
    ratios = []
    for row in summary.itertuples():
        if row.config in (V2, V2_CPU):
            ratios.append(row.qps_reported / base.get(
                (row.threads, row.queries, row.ef), np.nan))
        else:
            ratios.append(np.nan)
    return ratios


def flagged_rows(summary):
    return summary[summary["flag"] == 1]


def print_flagged(summary, host):
    flagged = flagged_rows(summary)
    one_launch = summary[(summary["launches"] < 2)
                         & (summary["config"] != V2_STAGE)]
    if not one_launch.empty:
        print("%s: %d configurations have one launch, so their spread is "
              "not measured" % (host, len(one_launch)))
    elif flagged.empty:
        print("%s: no configuration has launch spread above %g percent" % (
            host, FLAG_PERCENT))
    for row in flagged.itertuples():
        print("%s: FLAGGED %s, %d thread(s), %d queries, ef %d: launch "
              "medians %.0f to %.0f, reported %.0f, spread %.1f percent" % (
                  host, row.config, row.threads, row.queries, row.ef,
                  row.qps_launch_min, row.qps_launch_max, row.qps_reported,
                  row.spread_pct))


# ---------------------------------------------------------------- figures

def set_style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8, "axes.grid": True, "grid.color": "#E0E0E0",
        "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def pick(frame, config, threads, queries):
    """Rows of one configuration, thread count and query count, by ef."""
    mask = ((frame["config"] == config) & (frame["threads"] == threads)
            & (frame["queries"] == queries))
    return frame[mask].sort_values("ef")


def has_rows(frame, config, queries):
    return ((frame["config"] == config) & (frame["queries"] == queries)).any()


def plural(threads):
    return "%d thread%s" % (threads, "" if threads == 1 else "s")


def title_text(info, queries):
    host = info["host"] + (" (WSL2)" if info["platform"] == "wsl2" else "")
    gpu = info["gpu"] if info["gpu"] else "no GPU rows"
    if queries < info["max_queries"]:
        count = "first %d queries" % queries
    else:
        count = "%d queries" % queries
    return "%s, %s\n%s" % (host, gpu, count)


def save(fig, out_dir, name):
    for extension in ("png", "pdf"):
        # No creation time or software string in the files.
        meta = {"Software": None} if extension == "png" else {
            "CreationDate": None}
        path = os.path.join(out_dir, "%s.%s" % (name, extension))
        fig.savefig(path, dpi=200, metadata=meta)
    plt.close(fig)
    print("wrote %s.png and .pdf" % os.path.join(out_dir, name))


def finish(fig, caption, legend=None, columns=3, handler_map=None):
    """Caption at the bottom, an optional legend above it, then the layout.
    The figure grows by a line height for every caption line past three."""
    lines = textwrap.wrap(caption, 105, break_on_hyphens=False)
    fig.set_figheight(fig.get_figheight() + 0.125 * max(0, len(lines) - 3))
    height = fig.get_figheight()
    fig.text(0.01, 0.008, "\n".join(lines), ha="left", va="bottom",
             fontsize=7.5)
    bottom = (0.06 + 0.125 * len(lines)) / height
    if legend:
        handles, labels = legend
        rows = -(-len(handles) // columns)
        fig.legend(handles, labels, loc="lower center", ncol=columns,
                   bbox_to_anchor=(0.5, bottom), frameon=False,
                   handler_map=handler_map)
        bottom += (0.05 + 0.19 * rows) / height
    fig.tight_layout(rect=(0, bottom, 1, 1))


def thread_ratios(summary, version, queries):
    """qps_reported at 8 threads over that at 1 thread, by ef."""
    one = pick(summary, version, THREADS[0], queries).set_index("ef")
    many = pick(summary, version, THREADS[1], queries).set_index("ef")
    both = pd.concat([one["qps_reported"], many["qps_reported"]], axis=1,
                     join="inner")
    return both.iloc[:, 1] / both.iloc[:, 0]


def thread_note(summary, queries):
    """The sentence on V2 throughput against threads, or None unless the
    8-thread V2 qps is under twice the 1-thread qps at every ef."""
    v2 = thread_ratios(summary, V2, queries)
    if v2.empty or not (v2 < 2).all():
        return None
    text = "V2 at %d threads gave %.2f to %.2f times its 1-thread " \
           "throughput" % (THREADS[1], v2.min(), v2.max())
    v1 = thread_ratios(summary, V1, queries)
    if not v1.empty:
        text += " (V1: %.2f to %.2f times)" % (v1.min(), v1.max())
    return text + "."


def short_pass_note(summary, queries):
    """The note on V1 at 8 threads when its timed pass lasts milliseconds."""
    rows = pick(summary, V1, THREADS[1], queries)
    if queries >= SHORT_PASS_QUERIES or rows.empty:
        return None
    milliseconds = queries / rows["qps_reported"] * 1000
    return ("V1 at %d threads on %d queries is a pass of only %.1f to %.1f "
            "ms; QPS there reads 2 to 7 percent low." % (
                THREADS[1], queries, milliseconds.min(), milliseconds.max()))


def recall_qps_series(summary, search, queries):
    """The curves of the recall-QPS figure as (version, threads, summary
    rows, every repetition's row), the rows by ef."""
    series = []
    for version in (V1, V2):
        for threads in THREADS:
            rows = pick(summary, version, threads, queries)
            if not rows.empty:
                series.append((version, threads, rows,
                               pick(search, version, threads, queries)))
    return series


def plot_recall_qps(run, summary, queries, out_dir):
    info, search = run["info"], run["search"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    recall_of = {}
    for version, threads, rows, every in recall_qps_series(summary, search,
                                                           queries):
        style, marker = THREAD_STYLE[threads]
        color = VERSION_COLOR[version]
        ax.scatter(every["recall"], every["qps"], s=5, color=color,
                   alpha=0.3, linewidths=0, zorder=2)
        below = rows["qps_reported"] - rows["qps_row_min"]
        above = rows["qps_row_max"] - rows["qps_reported"]
        ax.errorbar(rows["recall"], rows["qps_reported"],
                    yerr=[below, above], color=color, linestyle=style,
                    marker=marker, markersize=4.5, linewidth=1.2,
                    elinewidth=0.9, capsize=2, zorder=3,
                    label="%s, %s" % (VERSION_LABEL[version],
                                      plural(threads)))
        for recall, ef in zip(rows["recall"], rows["ef"]):
            recall_of.setdefault(ef, recall)
    # One label per ef above the axes, on two levels so neighbors fit.
    for number, ef in enumerate(sorted(recall_of)):
        ax.axvline(recall_of[ef], color="#BBBBBB", linestyle=":",
                   linewidth=0.7, zorder=0)
        ax.text(recall_of[ef], 1.01 + 0.06 * (number % 2), "ef %d" % ef,
                transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8)
    ax.set_yscale("log")
    ax.margins(x=0.06, y=0.1)
    ax.set_xlabel("recall@%d (fraction of the true %d nearest found)" % (
        info["k"], info["k"]))
    ax.set_ylabel("throughput (queries per second, log scale)")
    ax.set_title(title_text(info, queries), pad=26)
    caption = ("Marker: median over launches of each launch's median "
               "repetition. Bar: lowest to highest over all rows (every "
               "repetition of every launch). Faint dots: every repetition.")
    if has_rows(summary, V2, queries):
        note = thread_note(summary, queries)
        if note:
            caption += " " + note
        else:
            print("%s: thread-count sentence left out of the recall_qps "
                  "caption: no V2 rows at both thread counts, or V2 at 8 "
                  "threads reaches twice its 1-thread throughput" %
                  info["host"])
    note = short_pass_note(summary, queries)
    if note:
        caption += " " + note
    finish(fig, caption, ax.get_legend_handles_labels(), columns=2)
    save(fig, out_dir, "recall_qps_%s_q%d" % (info["host"], queries))


def draw_stack(ax, xs, parts):
    """Stacked bars; parts are (values, color) from the bottom up."""
    bottom = np.zeros(len(xs))
    for values, color in parts:
        ax.bar(xs, values, bottom=bottom, color=color, width=0.72,
               edgecolor="white", linewidth=0.4)
        bottom = bottom + values
    return bottom


def stage_parts(rows):
    """Thread time per query by part for V2 throughput rows, bottom up."""
    inside = rows["gather_us"] + rows["submit_us"] + rows["wait_us"]
    other = (rows["backend_us"] - inside).clip(lower=0)
    return [(rows["outside_backend_us"].to_numpy(), STAGE_COLOR["outside"]),
            (rows["gather_us"].to_numpy(), STAGE_COLOR["gather"]),
            (rows["submit_us"].to_numpy(), STAGE_COLOR["submit"]),
            (rows["wait_us"].to_numpy(), STAGE_COLOR["wait"]),
            (other.to_numpy(), STAGE_COLOR["other"])]


def submit_share_range(search, threads, queries):
    """Lowest and highest percent of the backend time spent in submit, over
    the V2 throughput rows of one thread count; None without rows."""
    rows = pick(search, V2, threads, queries)
    if rows.empty:
        return None
    share = 100.0 * rows["submit_us"] / rows["backend_us"]
    return share.min(), share.max()


def repetition_note(search, queries):
    """Says that a bar is one repetition, and how far the submit share of a
    round trip moves between repetitions."""
    ranges = []
    for threads in THREADS:
        low_high = submit_share_range(search, threads, queries)
        if low_high is not None:
            ranges.append("%.0f to %.0f percent at %s" % (
                low_high + (plural(threads),)))
    text = ("Each bar is one repetition (the median-QPS repetition of the "
            "median launch)")
    if ranges:
        text += (("; across repetitions the submit share of a round trip "
                  "ranged from ") + " and from ".join(ranges))
    return text + "."


def thin_shares(rows, ef):
    """Percent of the bar of one ef taken by outside backend and by gather,
    or None without that ef."""
    rows = rows[rows["ef"] == ef]
    if rows.empty:
        return None
    values = [part[0][0] for part in stage_parts(rows)]
    total = sum(values)
    return 100.0 * values[0] / total, 100.0 * values[1] / total


def thin_note(summary, queries):
    """The shares of the two segments too thin to see in the stage figure."""
    texts = []
    for threads in THREADS:
        shares = thin_shares(pick(summary, V2, threads, queries),
                             OPERATING_EF)
        if shares is not None:
            texts.append("%s, outside backend %.1f%%, gather %.1f%% of the "
                         "bar" % ((plural(threads),) + shares))
    if not texts:
        return ""
    return (" At ef %d the outside-backend and gather segments are too thin "
            "to see: %s." % (OPERATING_EF, "; ".join(texts)))


def plot_stages(run, summary, queries, out_dir):
    info, search = run["info"], run["search"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.5, 4.4))
    for ax, threads in zip(axes, THREADS):
        rows = pick(summary, V2, threads, queries)
        ax.set_title("%s\nbar = latency of one query;\ntime per completed "
                     "query = bar / threads" % plural(threads), fontsize=8.5)
        ax.set_xlabel("ef")
        if rows.empty:
            ax.text(0.5, 0.5, "no rows", transform=ax.transAxes, ha="center")
            continue
        xs = np.arange(len(rows))
        totals = draw_stack(ax, xs, stage_parts(rows))
        for x, total in zip(xs, totals):
            ax.annotate("{:,.0f}".format(total), (x, total), ha="center",
                        fontsize=7, xytext=(0, 2),
                        textcoords="offset points")
        ax.set_xticks(xs, [str(ef) for ef in rows["ef"]])
        ax.margins(y=0.1)
        ax.yaxis.set_major_formatter("{x:,.0f}")
    axes[0].set_ylabel("thread time per query (microseconds)")
    fig.suptitle(title_text(info, queries), fontsize=9.5)
    handles = [
        Patch(color=STAGE_COLOR["outside"],
              label="outside backend (traversal + glue)"),
        Patch(color=STAGE_COLOR["gather"], label="gather"),
        Patch(color=STAGE_COLOR["other"], label="other inside backend"),
        Patch(color=STAGE_COLOR["submit"], label="submit"),
        Patch(color=STAGE_COLOR["wait"], label="wait"),
        Line2D([], [], color="none",
               label="(submit / wait: split varies with threads and "
                     "between repetitions)"),
    ]
    caption = ("V2 throughput runs. Number above a bar: total thread time "
               "per query (backend plus outside). %s%s" % (
                   repetition_note(search, queries),
                   thin_note(summary, queries)))
    ratio = round_trip_ratio(summary, queries)
    if ratio is not None and ratio >= QUEUE_RATIO:
        caption += (" In the %d-thread panel, wait is mostly time queued "
                    "behind other threads' round trips." % THREADS[1])
    else:
        print("%s: queueing sentence left out of the stages caption: a "
              "round trip does not cost %g times more thread time at %d "
              "threads than at %d" % (info["host"], QUEUE_RATIO, THREADS[1],
                                      THREADS[0]))
    finish(fig, caption, (handles, [h.get_label() for h in handles]), 2)
    save(fig, out_dir, "stages_%s_q%d" % (info["host"], queries))


def round_trip_ratio(summary, queries):
    """Thread time of a V2 round trip at 8 threads over that at 1 thread:
    backend_us per round trip of the throughput rows the round_trip figure
    draws, median over ef. None without rows at both thread counts."""
    cost = []
    for threads in THREADS:
        rows = pick(summary, V2, threads, queries)
        per_trip = (rows["backend_us"]
                    / rows["round_trips_per_query"]).dropna()
        if per_trip.empty:
            return None
        cost.append(per_trip.median())
    return cost[1] / cost[0]


def queueing_sentence(ratio):
    """The round_trip caption sentence on queueing, from round_trip_ratio."""
    if ratio >= QUEUE_RATIO:
        return ("At %d threads a round trip took %.1f times the thread time "
                "of a %d-thread round trip; wait and the stage times then "
                "include time queued behind other threads' round trips." % (
                    THREADS[1], ratio, THREADS[0]))
    return ("A round trip took about the same thread time at %d and %d "
            "threads (ratio %.1f)." % (THREADS + (ratio,)))


def per_round_trip(rows, column):
    """Microseconds per round trip: per-query stage time over round trips."""
    return (rows[column] / rows["round_trips_per_query"]).fillna(0).to_numpy()


def launch_floor(search, threads, queries):
    """(lowest, median, highest) over launches of launch_floor_us of the V2
    rows at one setting."""
    rows = search[search["config"].isin([V2, V2_STAGE])
                  & (search["threads"] == threads)
                  & (search["queries"] == queries)]
    per_launch = rows.drop_duplicates(["config", "launch"])["launch_floor_us"]
    return (float(per_launch.min()), float(per_launch.median()),
            float(per_launch.max()))


def kernel_is_mostly_floor(summary, search, threads, queries):
    """True if the kernel time per round trip of one panel is at most twice
    the highest launch floor."""
    rows = pick(summary, V2_STAGE, threads, queries)
    if rows.empty or rows["kernel_us"].isna().all():
        return False
    highest = launch_floor(search, threads, queries)[2]
    return not (per_round_trip(rows, "kernel_us") > 2 * highest).any()


def plot_round_trip(run, summary, queries, out_dir):
    info, search = run["info"], run["search"]
    efs = sorted(summary[(summary["config"] == V2)
                         & (summary["queries"] == queries)]["ef"].unique())
    n = len(efs)
    left, right = np.arange(n), n + 1 + np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6))
    for ax, threads in zip(axes, THREADS):
        low, floor, high = launch_floor(search, threads, queries)
        run_rows = pick(summary, V2, threads, queries).set_index("ef")
        stage_rows = pick(summary, V2_STAGE, threads, queries).set_index("ef")
        run_rows, stage_rows = run_rows.reindex(efs), stage_rows.reindex(efs)
        parts = [(per_round_trip(run_rows, c + "_us"), STAGE_COLOR[c])
                 for c in ("gather", "submit", "wait")]
        top = draw_stack(ax, left, parts)
        parts = [(per_round_trip(stage_rows, c), STAGE_COLOR[s])
                 for c, s in (("gather_us", "gather"),
                              ("to_device_us", "to_device"),
                              ("kernel_us", "kernel"),
                              ("from_device_us", "from_device"))
                 if stage_rows[c].notna().any()]
        top_stage = draw_stack(ax, right, parts) if parts else np.zeros(n)
        ax.axhspan(low, high, color="black", alpha=0.15, linewidth=0)
        ax.axhline(floor, color="black", linestyle=":", linewidth=1.1)
        ax.set_ylim(0, max(high, top.max(), top_stage.max()) * 1.3)
        ax.set_ylabel("microseconds per round trip (thread time)")
        ax.yaxis.set_major_formatter("{x:,.0f}")
        names = [("%d\n(%.0f)" % (ef, c)) if c == c else str(ef)
                 for ef, c in zip(efs, run_rows["candidates_per_round_trip"])]
        ax.set_xticks(np.concatenate([left, right]), names + names,
                      fontsize=7)
        ax.set_title(plural(threads), fontsize=9)
        ax.set_xlabel("ef (candidates per round trip)")
        ax.text(left.mean(), 0.96, "throughput runs", ha="center",
                va="top", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.text(right.mean(), 0.96, "stage-timing runs", ha="center",
                va="top", transform=ax.get_xaxis_transform(), fontsize=8)
    fig.suptitle(title_text(info, queries), fontsize=9.5)
    handles = [Patch(color=STAGE_COLOR[c], label=c.replace("_", " "))
               for c in ("gather", "submit", "wait", "to_device", "kernel",
                         "from_device")]
    floor_key = (Patch(color="black", alpha=0.15, linewidth=0),
                 Line2D([], [], color="black", linestyle=":", linewidth=1.1))
    labels = [h.get_label() for h in handles]
    handles.append(floor_key)
    labels.append("empty kernel launch + wait, one caller (range over "
                  "launches)")
    caption = ("Where host threads spent time. %s Stage-timing runs "
               "synchronize after every stage, so the two groups are "
               "separate measurements and are not additive. The two panels "
               "have different vertical scales." % repetition_note(
                   search, queries))
    ratio = round_trip_ratio(summary, queries)
    if ratio is not None:
        caption += " " + queueing_sentence(ratio)
    stage = summary[(summary["config"] == V2_STAGE)
                    & (summary["queries"] == queries)]
    if stage["gather_us"].isna().all():
        caption += " The stage-timing rows have no gather time."
        print("%s: stage-timing rows have no gather_us" % info["host"])
    mostly = []
    for threads in THREADS:
        if kernel_is_mostly_floor(summary, search, threads, queries):
            mostly.append(plural(threads))
        else:
            print("%s: kernel sentence left out for %s in the round_trip "
                  "caption, kernel exceeds twice the launch floor" % (
                      info["host"], plural(threads)))
    if mostly:
        caption += (" 'kernel' is mostly launch and synchronization cost at "
                    "this batch size (%s)." % " and ".join(mostly))
    finish(fig, caption, (handles, labels), 3, {tuple: HandlerTuple(None)})
    save(fig, out_dir, "round_trip_%s_q%d" % (info["host"], queries))


def operating_point(run, summary):
    """V2 at ef 64 on the most queries, or None: (candidates per round trip,
    microseconds per round trip, queries, (lowest, highest) microseconds per
    round trip over the 1-thread repetitions, largest batch or None).
    Candidates: median of the search.csv rows; time: the 1-thread summary
    line, which is the mean round trip of one repetition."""
    search = run["search"]
    rows = search[(search["config"] == V2) & (search["ef"] == OPERATING_EF)]
    if rows.empty:
        return None
    queries = int(rows["queries"].max())
    rows = rows[rows["queries"] == queries]
    one = pick(summary, V2, 1, queries)
    one = one[one["ef"] == OPERATING_EF]
    if one.empty:
        return None
    line = one.iloc[0]
    each = pick(search, V2, 1, queries)
    each = each[each["ef"] == OPERATING_EF]
    per_trip = each["backend_us"] / each["round_trips_per_query"]
    try:
        largest = int(float(rows["max_batch"].iloc[0]))
    except (KeyError, ValueError):
        largest = None
    return (rows["candidates_per_round_trip"].median(),
            line["backend_us"] / line["round_trips_per_query"], queries,
            (per_trip.min(), per_trip.max()), largest)


def plot_xfer(run, summary, out_dir):
    info, xfer = run["info"], run["xfer"]
    if xfer is None:
        return
    threads = int(xfer["cpu_threads"].iloc[0])
    runs = xfer["run"].nunique()
    lines = [  # column, label, color, style, marker
        ("ship_pinned_us", "ship pinned (vectors sent each time)",
         VERSION_COLOR[V2], "-", "o"),
        ("gather_us", "gather (part of ship)", STAGE_COLOR["gather"], "--",
         None),
        ("resident_us", "resident (only ids sent)", "#CC79A7", "-", "^"),
        ("cpu_1t_us", "CPU, 1 thread", VERSION_COLOR[V1], "-", "o"),
        ("cpu_all_us", "CPU, all %d threads" % threads, VERSION_COLOR[V1],
         "--", "s"),
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    by_size = xfer.groupby("pairs")
    for column, label, color, style, marker in lines:
        low, mid, high = (by_size[column].min(), by_size[column].median(),
                          by_size[column].max())
        ax.plot(mid.index, mid, color=color, linestyle=style, marker=marker,
                markersize=3.5, linewidth=1.2, label=label)
        ax.fill_between(mid.index, low, high, color=color, alpha=0.2,
                        linewidth=0)
    point = operating_point(run, summary)
    if point is not None:
        candidates, micro, _, (lowest, highest), largest = point
        ax.axvline(candidates, color=VERSION_COLOR[V2], linestyle=":",
                   linewidth=1)
        ax.plot([candidates], [micro], marker="*", markersize=11,
                color=VERSION_COLOR[V2], markeredgecolor="black",
                markeredgewidth=0.6, linestyle="none", zorder=5)
        limit = " (1 to %d)" % largest if largest else ""
        ax.annotate("V2 at ef %d, 1 thread: mean of one\nrepetition, %.0f "
                    "candidates on average%s,\n%.0f us; %.0f to %.0f us "
                    "across repetitions" % (
                        OPERATING_EF, candidates, limit, micro, lowest,
                        highest),
                    (candidates, micro), xytext=(0.07, 0.7),
                    textcoords="axes fraction", fontsize=8,
                    arrowprops={"arrowstyle": "-", "linewidth": 0.5})
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size (distance pairs per round trip)")
    ax.set_ylabel("microseconds per round trip")
    ax.set_title("%s, %s\nsingle caller" % (
        info["host"] + (" (WSL2)" if info["platform"] == "wsl2" else ""),
        info["gpu"]))
    caller = "Single caller"
    if info["platform"] == "wsl2":
        caller += "; under WSL2 the GPU also drives the desktop"
    caption = ("Line: median over %d run%s. Band: lowest to highest of those "
               "runs; a further run can fall outside it. %s. Pageable-memory "
               "shipping is measured but not drawn." % (
                   runs, "" if runs == 1 else "s", caller))
    handles, labels = ax.get_legend_handles_labels()
    finish(fig, caption, (handles, labels), 2)
    save(fig, out_dir, "xfer_%s" % info["host"])


def visits_title(info, cdf):
    """Title of the visits figure, with the number of learn queries."""
    title = "%s, k = %d: concentration of distance evaluations" % (
        info["dataset"], info["k"])
    whole = cdf[cdf["curve"] == "in-sample"]["queries"].dropna().unique()
    held = cdf[cdf["curve"] == "held-out"]["queries"].dropna().unique()
    if len(whole) == 1:
        title += "\n{:,.0f} learn queries".format(whole[0])
        if len(held) == 1:
            title += " (held-out curve: {:,.0f})".format(held[0])
    return title


def plot_visits(run, out_dir):
    info, cdf = run["info"], run["cdf"]
    if cdf is None:
        return
    curves = [  # curve, ef, label, color, style
        ("in-sample", 16, "in-sample, ef 16", "#E69F00", "-"),
        ("in-sample", 64, "in-sample, ef 64", "#0072B2", "-"),
        ("in-sample", 256, "in-sample, ef 256", "#CC79A7", "-"),
        ("held-out", 64, "held-out, ef 64", "#0072B2", "--"),
        ("uniform", 64, "uniform baseline, ef 64", "#0072B2", ":"),
    ]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Many points, so y = x is drawn as the curve it is on a log x axis.
    diagonal = np.logspace(-4, 0, 200)
    ax.plot(diagonal, diagonal, color="#999999", linewidth=0.6,
            label="y = x")
    for curve, ef, label, color, style in curves:
        rows = cdf[(cdf["curve"] == curve) & (cdf["ef"] == ef)]
        rows = rows.sort_values("share_of_vectors")
        if rows.empty:
            print("%s: no %s curve at ef %d in visit_cdf.csv" % (
                info["host"], curve, ef))
            continue
        ax.plot(rows["share_of_vectors"], rows["share_of_evaluations"],
                color=color, linestyle=style, linewidth=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("share of base vectors, most-visited first (log scale)")
    ax.set_ylabel("cumulative share of distance evaluations")
    ax.set_title(visits_title(info, cdf))
    note = "every query starts from the same entry point"
    if "entry_point" in cdf.columns and cdf["entry_point"].nunique() == 1:
        note += " (vector %s)" % cdf["entry_point"].iloc[0]
    ax.text(0.02, 0.97, note, transform=ax.transAxes, va="top", fontsize=8)
    caption = ("The counts come from the learn-set vectors used as queries, "
               "not from the timed queries. In-sample: vectors ranked by "
               "the counts being measured, so the curve is the most "
               "concentrated one possible. Held-out: ranked by the first "
               "half of the queries, counted on the second half. Uniform "
               "baseline: the same evaluations spread at random, ranked "
               "in-sample (compare with the in-sample curves). For the "
               "held-out curve the random baseline is y = x.")
    handles, labels = ax.get_legend_handles_labels()
    finish(fig, caption, (handles, labels), 3)
    save(fig, out_dir, "visits_%s" % info["host"])


# ----------------------------------------------------------------- tables

def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def qps_text(row):
    """QPS with the range over all rows. A star marks a launch spread above
    FLAG_PERCENT, a dagger a row range above ROW_MARK_PERCENT of the QPS."""
    if row is None:
        return "-"
    text = "%.0f (%.0f-%.0f)" % (row["qps_reported"], row["qps_row_min"],
                                 row["qps_row_max"])
    if row["flag"] == 1:
        text += "*"
    if row["row_spread_pct"] > ROW_MARK_PERCENT:
        text += DAGGER
    return text


def marks_note():
    return ("Marks: * launch spread above %g percent; %s range over all rows "
            "above %g percent of the reported value." % (
                FLAG_PERCENT, DAGGER, ROW_MARK_PERCENT))


def line_of(frame, ef):
    rows = frame[frame["ef"] == ef]
    return None if rows.empty else rows.iloc[0]


def table_a(summary, info, queries, threads):
    v1 = pick(summary, V1, threads, queries)
    v2 = pick(summary, V2, threads, queries)
    rows = []
    for ef in sorted(set(v1["ef"]) | set(v2["ef"])):
        a, b = line_of(v1, ef), line_of(v2, ef)
        recall = (a if a is not None else b)["recall"]
        ratio = "-" if b is None else "%.3g" % b["qps_vs_v1"]
        rows.append([str(ef), "%.4f" % recall, qps_text(a), qps_text(b),
                     ratio])
    title = "### %s, %d queries, %s" % (info["host"], queries,
                                        plural(threads))
    header = ["ef", "recall@%d" % info["k"],
              "V1 QPS (range over all rows)", "V2 QPS (range over all rows)",
              "V2 / V1"]
    text = [title, ""] + md_table(header, rows) + ["", marks_note(), ""]
    if threads == THREADS[1]:
        note = short_pass_note(summary, queries)
        if note:
            text += [note, ""]
    return text


def table_a2(summary, info, queries):
    """V1 alone on a query count that has no V2 rows."""
    one = pick(summary, V1, THREADS[0], queries)
    many = pick(summary, V1, THREADS[1], queries)
    rows = []
    for ef in sorted(set(one["ef"]) | set(many["ef"])):
        a, b = line_of(one, ef), line_of(many, ef)
        recall = (a if a is not None else b)["recall"]
        rows.append([str(ef), "%.4f" % recall, qps_text(a), qps_text(b)])
    title = "### %s, %d queries, V1 only" % (info["host"], queries)
    header = ["ef", "recall@%d" % info["k"],
              "V1 QPS, 1 thread (range over all rows)",
              "V1 QPS, 8 threads (range over all rows)"]
    text = [title, ""] + md_table(header, rows) + ["", marks_note(), ""]
    note = short_pass_note(summary, queries)
    if note:
        text += [note, ""]
    return text


def table_b(summary):
    rows = []
    cpu = summary[summary["config"] == V2_CPU]
    for _, row in cpu.iterrows():
        v1 = line_of(pick(summary, V1, row["threads"], row["queries"]),
                     row["ef"])
        rows.append([str(row["queries"]), str(row["threads"]),
                     str(row["ef"]), qps_text(v1), qps_text(row),
                     "%.3g" % row["qps_vs_v1"]])
    if not rows:
        return ["No rows of the V2 driver with CPU distances.", ""]
    header = ["queries", "threads", "ef", "V1 QPS (range over all rows)",
              "V2 driver, CPU distances QPS (range over all rows)", "ratio"]
    text = md_table(header, rows) + ["", marks_note(), ""]
    for queries in sorted(cpu[cpu["threads"] == THREADS[1]]["queries"]
                          .unique()):
        note = short_pass_note(summary, queries)
        if note:
            text += [note, ""]
    return text


def curve_name(curve, ef):
    return "%s, ef %d" % (curve, ef)


def table_c(cdf):
    header = ["curve"] + ["top %g%%" % (100 * s) for s in SHARES]
    rows = []
    order = {"in-sample": 0, "held-out": 1, "uniform": 2}
    groups = sorted(cdf.groupby(["curve", "ef"]),
                    key=lambda g: (order.get(g[0][0], 3), g[0][1]))
    for (curve, ef), group in groups:
        group = group.sort_values("share_of_vectors")
        got = np.interp(SHARES, group["share_of_vectors"],
                        group["share_of_evaluations"])
        rows.append([curve_name(curve, int(ef))] +
                    ["%.1f%%" % (100 * g) for g in got])
    return md_table(header, rows) + [
        "", "Share of all distance evaluations received by the most-visited "
        "share of base vectors, read from visit_cdf.csv (linear "
        "interpolation between its points).", ""]


def extra_round_lines(summary):
    """One line for each configuration with more launches than the fewest
    of any configuration, or nothing when all have as many."""
    fewest = summary["launches"].min()
    more = summary[summary["launches"] > fewest].groupby(
        ["config", "threads", "queries"], sort=False)["launches"].max()
    if more.empty:
        return []
    lines = ["Configurations with more launches than the others:", ""]
    for (config, threads, queries), launches in more.items():
        lines.append("- %s, %s, %d queries: %d launches instead of %d; one "
                     "extra round was run; see launches.log for the reason." %
                     (config, plural(threads), queries, launches, fewest))
    return lines + [""]


def table_d(summary):
    flagged = flagged_rows(summary)
    single = summary[(summary["launches"] < 2)
                     & (summary["config"] != V2_STAGE)]
    text = []
    if not single.empty:
        text = ["%d configurations have one launch, so their spread is not "
                "measured." % len(single), ""]
    text += extra_round_lines(summary)
    if flagged.empty:
        text += ["No configuration has launch spread above %g percent." %
                 FLAG_PERCENT, ""]
    else:
        rows = [[r.config, str(r.threads), str(r.queries), str(r.ef),
                 str(r.launches), "%.0f" % r.qps_reported,
                 "%.0f-%.0f" % (r.qps_launch_min, r.qps_launch_max),
                 "%.1f" % r.spread_pct, "%.1f" % r.row_spread_pct]
                for r in flagged.itertuples()]
        header = ["config", "threads", "queries", "ef", "launches",
                  "QPS reported", "launch medians (min-max)", "spread %",
                  "row range %"]
        text += md_table(header, rows) + [""]
    text += ["### Unflagged but variable between repetitions", ""]
    variable = summary[(summary["flag"] != 1)
                       & (summary["row_spread_pct"] > ROW_LIST_PERCENT)]
    if variable.empty:
        text += ["No unflagged configuration has a row range above %g "
                 "percent of its reported QPS." % ROW_LIST_PERCENT, ""]
        return text
    rows = [[r.config, str(r.threads), str(r.queries), str(r.ef),
             str(r.launches), "%.0f" % r.qps_reported,
             "%.0f-%.0f" % (r.qps_row_min, r.qps_row_max),
             "%.1f" % r.row_spread_pct] for r in variable.itertuples()]
    header = ["config", "threads", "queries", "ef", "launches",
              "QPS reported", "all rows (min-max)", "row range %"]
    text += ["Not flagged by launch spread, but the range over all rows "
             "exceeds %g percent of the reported QPS." % ROW_LIST_PERCENT, ""]
    return text + md_table(header, rows) + [""]


def write_tables(run, summary, path):
    info = run["info"]
    text = ["# Tables for %s" % info["host"], "",
            "QPS is the median over launches of each launch's median "
            "repetition. The range in parentheses runs from the lowest to "
            "the highest QPS over all rows (every repetition of every "
            "launch), so it includes the variation between repetitions. A "
            "star marks a launch spread (highest launch median minus lowest, "
            "over the reported QPS) above %g percent; a dagger marks a row "
            "range above %g percent of the reported QPS." % (
                FLAG_PERCENT, ROW_MARK_PERCENT), ""]
    if info["platform"] == "wsl2":
        text += ["This host runs under WSL2.", ""]
    text += ["## (a) V1 against V2, by ef", ""]
    for queries in sorted(summary["queries"].unique()):
        if not has_rows(summary, V2, queries):
            continue
        for threads in THREADS:
            text += table_a(summary, info, queries, threads)
    only_v1 = [q for q in sorted(summary["queries"].unique())
               if q == info["max_queries"] and has_rows(summary, V1, q)
               and not has_rows(summary, V2, q)]
    if only_v1:
        text += ["## (a2) V1 on the full query set", ""]
        for queries in only_v1:
            text += table_a2(summary, info, queries)
    text += ["## (b) V2 driver with CPU distances against V1", ""]
    text += table_b(summary)
    text += ["## (c) Concentration of distance evaluations", ""]
    if run["cdf"] is None:
        text += ["No visit_cdf.csv in the results directory.", ""]
    else:
        text += table_c(run["cdf"])
    text += ["## (d) Flagged configurations", ""] + table_d(summary)
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(text))
    print("wrote %s" % path)


# ------------------------------------------------------------------- main

def report(run, out_root):
    info = run["info"]
    out_dir = os.path.join(out_root, info["host"])
    os.makedirs(out_dir, exist_ok=True)
    summary = summarize(run["search"])
    path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(path, index=False, float_format="%.6g")
    print("wrote %s (%d lines)" % (path, len(summary)))
    print_flagged(summary, info["host"])
    for queries in sorted(summary["queries"].unique()):
        plot_recall_qps(run, summary, queries, out_dir)
        if has_rows(summary, V2, queries):
            plot_stages(run, summary, queries, out_dir)
            plot_round_trip(run, summary, queries, out_dir)
    plot_xfer(run, summary, out_dir)
    plot_visits(run, out_dir)
    write_tables(run, summary, os.path.join(out_dir, "tables.md"))


def main():
    parser = argparse.ArgumentParser(
        description="Summary, tables and figures from benchmark CSVs.")
    parser.add_argument("results", nargs="+",
                        help="results directory of one machine")
    parser.add_argument("--out", required=True, help="figure directory")
    args = parser.parse_args()

    runs = [load_directory(directory) for directory in args.results]
    hosts = [r["info"]["host"] for r in runs if r is not None]
    for host in sorted(set(hosts)):
        if hosts.count(host) > 1:
            problem("host %s is in more than one results directory" % host)
    refuse_if_problems()
    set_style()
    for run in runs:
        report(run, args.out)


if __name__ == "__main__":
    main()
