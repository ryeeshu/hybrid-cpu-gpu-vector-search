#!/usr/bin/env python3
"""Tests of tools/plot.py, tools/check_results.py and tools/visit_cdf.py on
small hand-written fixtures.

Usage: test_tools.py SCRATCH_DIR [--tools DIR]
SCRATCH_DIR is emptied first and must be unique to this run. The tools come
from DIR, from $HVS_TOOLS_DIR, or from ../tools, in that order. The plot.py
tests need pandas and matplotlib; without them the program prints SKIPPED for
that part and runs the rest.
"""

import argparse
import csv
import importlib.util
import io
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

sys.dont_write_bytecode = True  # no __pycache__ next to the tools

TOOLS = ""
SCRATCH = ""

# Columns of search.csv, as written by hvs_search.
HEADER = (
    "build_id tag host platform build compiler build_type cxx_flags "
    "logical_cpus omp_proc_bind omp_places dataset_hash n d degree queries k "
    "ef mode threads threads_used launch repetition warmups recall qps "
    "latency_mean_us latency_p50_us latency_p95_us latency_p99_us "
    "evals_per_query hops_per_query batches_per_query result_hash backend "
    "stage_timing verified gpu cuda_runtime cuda_driver wait_policy "
    "max_batch pinned resident init_s device_bytes launch_floor_us "
    "round_trips_per_query candidates_per_round_trip "
    "bytes_to_device_per_query bytes_from_device_per_query backend_us "
    "outside_backend_us gather_us submit_us wait_us to_device_us kernel_us "
    "from_device_us start_time_utc command").split()
BUILD_ID = "0123456789ab"
DATASET_HASH = "feedfacefeedface"


def scratch_dir(name):
    path = os.path.join(SCRATCH, name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def run_python(script, *args):
    """Runs a script of the tools; returns (exit status, stdout, stderr)."""
    done = subprocess.run([sys.executable, os.path.join(TOOLS, script)] +
                          [str(a) for a in args], capture_output=True,
                          text=True)
    return done.returncode, done.stdout, done.stderr


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ plot.py

# QPS of the nine rows of each group at ef 64, as {launch: (repetition 1, 2,
# 3)}. Medians differ from means, and the median launch is not always launch
# 1 or 3. Median repetition of each launch (and which repetition it is), then
# the median launch: its value, launch and repetition, the row the stage
# columns come from:
#   V1, 1 thread:  110 (1), 200 (3), 160 (3)    -> 160, launch 3, rep 3
#   V1, 8 threads: 1000 (1), 1400 (3), 1900 (3) -> 1400, launch 2, rep 3
#   V2, 1 thread:  80 (1), 50 (1), 65 (3)       -> 65, launch 3, rep 3
#   V2, 8 threads: 80 (1), 100 (2), 115 (3)     -> 100, launch 2, rep 2
#   stage, 1 thread:  75 (3), 79 (1), 90 (1)    -> 79, launch 2, rep 1
#   stage, 8 threads: 300 (1), 390 (3), 340 (3) -> 340, launch 3, rep 3
# The mean of the launch medians of V1 at 1 thread is 156.7, and its launch
# means are 110, 180 and 236.7, so a mean in place of a median shows.
QPS64 = {
    ("V1", 1): {1: (110, 100, 120), 2: (210, 130, 200), 3: (150, 400, 160)},
    ("V1", 8): {1: (1000, 900, 1400), 2: (1500, 1300, 1400),
                3: (1800, 2000, 1900)},
    ("V2", 1): {1: (80, 70, 90), 2: (50, 60, 40), 3: (55, 95, 65)},
    ("V2", 8): {1: (80, 90, 70), 2: (95, 100, 105), 3: (120, 110, 115)},
    ("V2 stage-timing", 1): {1: (70, 80, 75), 2: (79, 82, 78),
                             3: (90, 85, 95)},
    ("V2 stage-timing", 8): {1: (300, 310, 290), 2: (400, 380, 390),
                             3: (350, 330, 340)},
}
# ef 16 has these multiples of the QPS of ef 64.
SCALE16 = {("V1", 1): 2, ("V1", 8): 3.2, ("V2", 1): 2, ("V2", 8): 2.5,
           ("V2 stage-timing", 1): 1.5, ("V2 stage-timing", 8): 1.5}
# V1 on a longer query set: all launches have the same three repetitions.
QPS_FULL = {1: (150, 150, 150), 8: (900, 1000, 1100)}
FULL_QUERIES = 200
SHORT_QUERIES = 100
RECALL = {(SHORT_QUERIES, 16): "0.800000", (SHORT_QUERIES, 64): "0.960000",
          (FULL_QUERIES, 64): "0.970000"}
RESULT_HASH = {(SHORT_QUERIES, 16): "1111111111111111",
               (SHORT_QUERIES, 64): "2222222222222222",
               (FULL_QUERIES, 64): "3333333333333333"}
# Batches per query and candidates per round trip.
BATCHES = {16: 21, 64: 63}
CANDIDATES = {16: 20, 64: 18}


def fixture_row(config, threads, queries, ef, launch, repetition, qps):
    """One row of search.csv. latency_p50_us holds 10 * launch + repetition,
    so a summary line shows which row its stage columns came from."""
    tag = 10 * launch + repetition
    row = dict.fromkeys(HEADER, "")
    row.update(
        build_id=BUILD_ID, tag="t", host="TESTBOX", platform="linux",
        build="release", compiler="gcc", build_type="Release",
        cxx_flags="-O3", logical_cpus="16", omp_proc_bind="close",
        omp_places="cores", dataset_hash=DATASET_HASH, n="1000000", d="128",
        degree="32", queries=queries, k=10, ef=ef, threads=threads,
        threads_used=threads, launch=launch, repetition=repetition,
        warmups=1, recall=RECALL[(queries, ef)], qps="%g" % qps,
        latency_mean_us="50", latency_p50_us=tag, latency_p95_us="60",
        latency_p99_us="70", evals_per_query="400", hops_per_query="20",
        batches_per_query=BATCHES[ef],
        result_hash=RESULT_HASH[(queries, ef)], verified="no")
    if config == "V1":
        row.update(mode="cpu")
        return row
    stage = config == "V2 stage-timing"
    # Backend plus outside time is the latency of 1000 microseconds; with 8
    # threads the backend share is smaller.
    backend = (700 if threads == 1 else 400) + tag
    row.update(
        mode="hybrid-sync", backend="cuda", stage_timing="1" if stage else "0",
        gpu="Test GPU", cuda_runtime="12.6", cuda_driver="13.0",
        wait_policy="auto", max_batch="32", pinned="1", resident="0",
        launch_floor_us=20 + 5 * launch,
        round_trips_per_query=BATCHES[ef],
        candidates_per_round_trip=CANDIDATES[ef], latency_mean_us="1000",
        backend_us=backend, outside_backend_us=1000 - backend,
        gather_us=tag)
    if stage:
        row.update(to_device_us=100 + tag,
                   kernel_us=300 if threads == 1 else 3000,
                   from_device_us=200)
    else:
        row.update(submit_us=100 + 3 * tag, wait_us=100 if threads == 8
                   else 300)
    return row


def fixture_rows():
    rows = []
    for (config, threads), by_launch in QPS64.items():
        for ef, scale in ((16, SCALE16[(config, threads)]), (64, 1)):
            for launch, qps_list in by_launch.items():
                for repetition, qps in enumerate(qps_list, 1):
                    rows.append(fixture_row(
                        config, threads, SHORT_QUERIES, ef, launch,
                        repetition, qps * scale))
    for threads, qps_list in QPS_FULL.items():
        for launch in (1, 2, 3):
            for repetition, qps in enumerate(qps_list, 1):
                rows.append(fixture_row("V1", threads, FULL_QUERIES, 64,
                                        launch, repetition, qps))
    return rows


def write_search_csv(directory, rows):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "search.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


# Summary line of each group at ef 64: reported QPS, lowest and highest launch
# median, lowest and highest of all rows, launch and repetition of the row
# the stage columns come from.
EXPECTED64 = {
    ("V1", 1): (160, 110, 200, 100, 400, 3, 3),
    ("V1", 8): (1400, 1000, 1900, 900, 2000, 2, 3),
    ("V2", 1): (65, 50, 80, 40, 95, 3, 3),
    ("V2", 8): (100, 80, 115, 70, 120, 2, 2),
}
# Stage-timing groups: the row the stage columns come from.
STAGE_DETAIL64 = {("V2 stage-timing", 1): (2, 1),
                  ("V2 stage-timing", 8): (3, 3)}


class PlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot = load_module("plot", os.path.join(TOOLS, "plot.py"))
        cls.directory = scratch_dir("plot_fixture")
        cls.rows = fixture_rows()
        write_search_csv(cls.directory, cls.rows)
        with redirect_stdout(io.StringIO()):
            cls.data = cls.plot.load_directory(cls.directory)
        if cls.plot.problems:
            raise AssertionError("fixture refused: %s" % cls.plot.problems)
        cls.search = cls.data["search"]
        cls.summary = cls.plot.summarize(cls.search)

    def line(self, config, threads, queries=SHORT_QUERIES, ef=64):
        rows = self.summary[(self.summary["config"] == config)
                            & (self.summary["threads"] == threads)
                            & (self.summary["queries"] == queries)
                            & (self.summary["ef"] == ef)]
        self.assertEqual(len(rows), 1, (config, threads, queries, ef))
        return rows.iloc[0]

    def test_summary_values(self):
        for (config, threads), want in EXPECTED64.items():
            line = self.line(config, threads)
            names = ["qps_reported", "qps_launch_min", "qps_launch_max",
                     "qps_row_min", "qps_row_max", "detail_launch",
                     "detail_repetition"]
            for name, value in zip(names, want):
                self.assertAlmostEqual(line[name], value, places=6,
                                       msg="%s %d %s" % (config, threads,
                                                         name))
            self.assertEqual(line["rows"], 9)
            self.assertEqual(line["launches"], 3)

    def test_summary_scaled_ef(self):
        # ef 16 is a multiple of ef 64, so the same rows are the medians.
        for (config, threads), want in EXPECTED64.items():
            line = self.line(config, threads, ef=16)
            self.assertAlmostEqual(line["qps_reported"],
                                   want[0] * SCALE16[(config, threads)])
            self.assertEqual((line["detail_launch"],
                              line["detail_repetition"]), want[5:])

    def test_stage_columns_come_from_the_detail_row(self):
        # Every row's latency_p50_us and gather_us are 10 * launch +
        # repetition, to_device_us is 100 more.
        for (config, threads), (launch, repetition) in (
                list(STAGE_DETAIL64.items()) + [
                    (k, v[5:]) for k, v in EXPECTED64.items()]):
            line = self.line(config, threads)
            tag = 10 * launch + repetition
            self.assertEqual((line["detail_launch"],
                              line["detail_repetition"]),
                             (launch, repetition), (config, threads))
            self.assertEqual(line["latency_p50_us"], tag)
            if config != "V1":
                self.assertEqual(line["gather_us"], tag)
            if config == "V2 stage-timing":
                self.assertEqual(line["to_device_us"], 100 + tag)

    def test_stage_timing_rows_have_no_qps(self):
        stage = self.summary[self.summary["config"] == "V2 stage-timing"]
        self.assertEqual(len(stage), 4)
        for name in ("qps_reported", "qps_launch_min", "qps_launch_max",
                     "qps_row_min", "qps_row_max", "spread_pct", "flag",
                     "row_spread_pct", "qps_vs_v1"):
            self.assertTrue(stage[name].isna().all(), name)

    def test_ratio_to_v1_uses_the_same_thread_count(self):
        self.assertAlmostEqual(self.line("V2", 1)["qps_vs_v1"], 65 / 160)
        self.assertAlmostEqual(self.line("V2", 8)["qps_vs_v1"], 100 / 1400)
        self.assertAlmostEqual(self.line("V2", 8, ef=16)["qps_vs_v1"],
                               250 / 4480)
        self.assertTrue(math.isnan(self.line("V1", 8)["qps_vs_v1"]))

    def test_spread_columns(self):
        line = self.line("V2", 8)
        self.assertAlmostEqual(line["spread_pct"], 35.0)
        self.assertEqual(line["flag"], 1)
        self.assertAlmostEqual(line["row_spread_pct"], 50.0)
        line = self.line("V1", 8, queries=FULL_QUERIES)
        self.assertEqual(line["flag"], 0)
        self.assertAlmostEqual(line["row_spread_pct"], 20.0)

    def test_recall_qps_rows_are_one_configuration(self):
        series = self.plot.recall_qps_series(self.summary, self.search,
                                             SHORT_QUERIES)
        self.assertEqual([(v, t) for v, t, _, _ in series],
                         [("V1", 1), ("V1", 8), ("V2", 1), ("V2", 8)])
        for version, threads, rows, every in series:
            self.assertEqual(set(rows["config"]), {version})
            self.assertEqual(set(every["config"]), {version})
            self.assertEqual(set(rows["threads"]), {threads})
            self.assertEqual(list(rows["ef"]), [16, 64])
            self.assertEqual(len(every), 18)

    def test_thread_note(self):
        want = ("V2 at 8 threads gave 1.54 to 1.92 times its 1-thread "
                "throughput (V1: 8.75 to 14.00 times).")
        self.assertEqual(self.plot.thread_note(self.summary, SHORT_QUERIES),
                         want)
        # No sentence when V2 at 8 threads reaches twice its 1-thread QPS.
        faster = self.summary.copy()
        mask = (faster["config"] == "V2") & (faster["threads"] == 8)
        faster.loc[mask, "qps_reported"] *= 2
        self.assertIsNone(self.plot.thread_note(faster, SHORT_QUERIES))

    def test_short_pass_note(self):
        want = ("V1 at 8 threads on 100 queries is a pass of only 22.3 to "
                "71.4 ms; QPS there reads 2 to 7 percent low.")
        self.assertEqual(self.plot.short_pass_note(self.summary,
                                                   SHORT_QUERIES), want)
        long_run = self.summary.copy()
        long_run["queries"] = long_run["queries"].replace(
            SHORT_QUERIES, self.plot.SHORT_PASS_QUERIES)
        self.assertIsNone(self.plot.short_pass_note(
            long_run, self.plot.SHORT_PASS_QUERIES))

    def test_launch_floor_range(self):
        # launch_floor_us is 20 + 5 * launch.
        self.assertEqual(self.plot.launch_floor(self.search, 1,
                                                SHORT_QUERIES),
                         (25.0, 30.0, 35.0))

    def test_submit_share_range(self):
        for threads in (1, 8):
            shares = [100.0 * float(r["submit_us"]) / float(r["backend_us"])
                      for r in self.rows
                      if r["mode"] == "hybrid-sync"
                      and r["stage_timing"] == "0"
                      and int(r["threads"]) == threads
                      and int(r["queries"]) == SHORT_QUERIES]
            low, high = self.plot.submit_share_range(self.search, threads,
                                                     SHORT_QUERIES)
            self.assertAlmostEqual(low, min(shares))
            self.assertAlmostEqual(high, max(shares))

    def test_operating_point(self):
        point = self.plot.operating_point(self.data, self.summary)
        candidates, micro, queries, (low, high), largest = point
        self.assertEqual((candidates, queries, largest), (18, 100, 32))
        # backend_us is 700 + 10 * launch + repetition at 1 thread; the
        # detail row is launch 3, repetition 3.
        self.assertAlmostEqual(micro, 733 / 63)
        self.assertAlmostEqual(low, 711 / 63)
        self.assertAlmostEqual(high, 733 / 63)

    def test_figures_and_tables(self):
        captions = []
        original = self.plot.finish

        def record(fig, caption, *args, **kwargs):
            captions.append(caption)
            return original(fig, caption, *args, **kwargs)

        out = scratch_dir("plot_out")
        self.plot.finish = record
        try:
            with redirect_stdout(io.StringIO()):
                self.plot.report(self.data, out)
        finally:
            self.plot.finish = original
        host_dir = os.path.join(out, "TESTBOX")
        self.check_captions(captions)
        self.check_tables(os.path.join(host_dir, "tables.md"))
        with open(os.path.join(host_dir, "summary.csv")) as f:
            columns = next(csv.reader(f))
        self.assertIn("row_spread_pct", columns)
        for name in ("recall_qps_TESTBOX_q100", "stages_TESTBOX_q100",
                     "round_trip_TESTBOX_q100"):
            self.assertTrue(os.path.getsize(
                os.path.join(host_dir, name + ".png")) > 0)

    def percent_text(self, numbers):
        return "%.0f to %.0f percent" % numbers

    def submit_range(self, threads):
        shares = [100.0 * float(r["submit_us"]) / float(r["backend_us"])
                  for r in self.rows
                  if r["mode"] == "hybrid-sync" and r["stage_timing"] == "0"
                  and int(r["threads"]) == threads
                  and int(r["queries"]) == SHORT_QUERIES]
        return min(shares), max(shares)

    def thin_text(self, threads):
        """Outside-backend and gather share of the ef 64 bar, from the raw
        row the bar is drawn from."""
        launch, repetition = EXPECTED64[("V2", threads)][5:]
        row = next(r for r in self.rows if r["mode"] == "hybrid-sync"
                   and r["stage_timing"] == "0"
                   and int(r["threads"]) == threads
                   and int(r["queries"]) == SHORT_QUERIES
                   and int(r["ef"]) == 64 and int(r["launch"]) == launch
                   and int(r["repetition"]) == repetition)
        outside, gather, submit, wait, backend = (
            float(row[c]) for c in ("outside_backend_us", "gather_us",
                                    "submit_us", "wait_us", "backend_us"))
        other = max(0.0, backend - gather - submit - wait)
        total = outside + gather + submit + wait + other
        return "outside backend %.1f%%, gather %.1f%% of the bar" % (
            100 * outside / total, 100 * gather / total)

    def check_captions(self, captions):
        by_name = {}
        for caption in captions:
            if caption.startswith("Marker"):
                by_name.setdefault("recall", []).append(caption)
            elif caption.startswith("V2 throughput runs"):
                by_name["stages"] = caption
            elif caption.startswith("Where host threads"):
                by_name["round_trip"] = caption
        recall, stages, trip = (by_name["recall"][0], by_name["stages"],
                                by_name["round_trip"])
        self.assertEqual(len(by_name["recall"]), 2)
        self.assertIn("Bar: lowest to highest over all rows", recall)
        self.assertIn("V2 at 8 threads gave 1.54 to 1.92 times its 1-thread "
                      "throughput (V1: 8.75 to 14.00 times).", recall)
        self.assertIn("is a pass of only 22.3 to 71.4 ms; QPS there reads 2 "
                      "to 7 percent low.", recall)
        self.assertNotIn("gave", by_name["recall"][1])
        for caption in (stages, trip):
            self.assertIn("Each bar is one repetition", caption)
            self.assertIn("across repetitions the submit share of a round "
                          "trip ranged from %s at 1 thread and from %s at 8 "
                          "threads" % (
                              self.percent_text(self.submit_range(1)),
                              self.percent_text(self.submit_range(8))),
                          caption)
        self.assertIn("1 thread, %s; 8 threads, %s." % (
            self.thin_text(1), self.thin_text(8)), stages)
        # A round trip costs 422/733 = 0.58 times as much at 8 threads as at
        # 1 (the same ratio at every ef), so neither queueing sentence.
        self.assertNotIn("queued", stages)
        self.assertNotIn("queued", trip)
        self.assertIn("A round trip took about the same thread time at 1 and "
                      "8 threads (ratio 0.6).", trip)
        # Kernel time is under twice the highest floor at 1 thread only.
        self.assertTrue(trip.endswith("at this batch size (1 thread)."), trip)

    def check_tables(self, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("V2 QPS (range over all rows)", text)
        self.assertIn("| 64 | 0.9600 | 160 (100-400)*† | "
                      "65 (40-95)*† | 0.406 |", text)
        self.assertIn("| 64 | 0.9600 | 1400 (900-2000)*† | "
                      "100 (70-120)*† | 0.0714 |", text)
        self.assertIn("Marks: * launch spread above 5 percent; † range "
                      "over all rows above 10 percent", text)
        self.assertIn("V1 at 8 threads on 100 queries is a pass of only",
                      text)
        # Table (a2): V1 on the longer query set.
        a2 = text.split("## (a2)")[1].split("## (b)")[0]
        self.assertIn("| 64 | 0.9700 | 150 (150-150) | 1000 (900-1100)† |",
                      a2)
        # Table (d): V1 at 8 threads on 200 queries is not flagged but its
        # rows range over 20 percent; at 1 thread they do not vary.
        d_text = text.split("## (d)")[1]
        flagged, other = d_text.split("### Unflagged but variable between "
                                      "repetitions")
        self.assertIn("| V2 | 8 | 100 | 64 | 3 | 100 | 80-115 | 35.0 | 50.0 |",
                      flagged)
        self.assertIn("| V1 | 8 | 200 | 64 | 3 | 1000 | 900-1100 | 20.0 |",
                      other)
        self.assertNotIn("| V1 | 1 | 200 |", other)
        self.assertNotIn("| V1 | 1 | 200 |", flagged)
        self.assertNotIn("one extra round was run", text)

    def captions(self, summary):
        """The captions of the stages and round_trip figures of a summary."""
        found = []
        original = self.plot.finish

        def record(fig, caption, *args, **kwargs):
            found.append(caption)
            return original(fig, caption, *args, **kwargs)

        out = scratch_dir("plot_captions")
        self.plot.finish = record
        try:
            with redirect_stdout(io.StringIO()):
                self.plot.plot_stages(self.data, summary, SHORT_QUERIES, out)
                self.plot.plot_round_trip(self.data, summary, SHORT_QUERIES,
                                          out)
        finally:
            self.plot.finish = original
        return found

    def test_queueing_sentences_follow_the_round_trip_ratio(self):
        mask = ((self.summary["config"] == "V2")
                & (self.summary["threads"] == 8))
        base = self.plot.round_trip_ratio(self.summary, SHORT_QUERIES)
        self.assertAlmostEqual(base, 422 / 733)
        for target, queued in ((1.4, False), (1.6, True), (4.0, True)):
            summary = self.summary.copy()
            summary.loc[mask, "backend_us"] *= target / base
            stages, trip = self.captions(summary)
            self.assertEqual("queued behind" in stages, queued, stages)
            if queued:
                self.assertIn(
                    "At 8 threads a round trip took %.1f times the thread "
                    "time of a 1-thread round trip; wait and the stage times "
                    "then include time queued behind other threads' round "
                    "trips." % target, trip)
                self.assertIn("In the 8-thread panel, wait is mostly time "
                              "queued behind other threads' round trips.",
                              stages)
            else:
                self.assertIn("A round trip took about the same thread time "
                              "at 1 and 8 threads (ratio %.1f)." % target,
                              trip)
                self.assertNotIn("queued behind", trip)


# Launch 4 of V2 at 8 threads, three repetitions at ef 64. With launch
# 1 to 3 (medians 80, 100 and 115) the launch medians are 80, 100, 105 and 115,
# so the reported QPS is (100 + 105) / 2 and the summary line takes its stage
# columns from launch 2, the lower of the two middle ones.
EXTRA_QPS64 = (105, 106, 104)


class ExtraLaunchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plot = load_module("plot_extra", os.path.join(TOOLS, "plot.py"))
        cls.directory = scratch_dir("plot_extra_fixture")
        rows = fixture_rows()
        for ef, scale in ((16, SCALE16[("V2", 8)]), (64, 1)):
            for repetition, qps in enumerate(EXTRA_QPS64, 1):
                rows.append(fixture_row("V2", 8, SHORT_QUERIES, ef, 4,
                                        repetition, qps * scale))
        write_search_csv(cls.directory, rows)
        with redirect_stdout(io.StringIO()):
            cls.data = cls.plot.load_directory(cls.directory)
        if cls.plot.problems:
            raise AssertionError("fixture refused: %s" % cls.plot.problems)
        cls.summary = cls.plot.summarize(cls.data["search"])

    def line(self, config, threads, ef=64):
        rows = self.summary[(self.summary["config"] == config)
                            & (self.summary["threads"] == threads)
                            & (self.summary["queries"] == SHORT_QUERIES)
                            & (self.summary["ef"] == ef)]
        self.assertEqual(len(rows), 1)
        return rows.iloc[0]

    def test_four_launches_are_the_mean_of_the_middle_two(self):
        line = self.line("V2", 8)
        self.assertAlmostEqual(line["qps_reported"], 102.5)
        self.assertEqual((line["qps_launch_min"], line["qps_launch_max"]),
                         (80, 115))
        self.assertEqual((line["qps_row_min"], line["qps_row_max"]),
                         (70, 120))
        self.assertEqual((line["launches"], line["rows"]), (4, 12))
        self.assertEqual((line["detail_launch"], line["detail_repetition"]),
                         (2, 2))
        self.assertAlmostEqual(line["spread_pct"], 100 * 35 / 102.5)
        self.assertAlmostEqual(self.line("V2", 8, ef=16)["qps_reported"],
                               2.5 * 102.5)
        self.assertAlmostEqual(line["qps_vs_v1"], 102.5 / 1400)

    def test_other_configurations_keep_three_launches(self):
        for config, threads in (("V1", 1), ("V1", 8), ("V2", 1)):
            self.assertEqual(self.line(config, threads)["launches"], 3)
        self.assertEqual(self.line("V2", 1)["qps_reported"], 65)

    def test_table_d_lists_the_configuration_with_an_extra_round(self):
        path = os.path.join(scratch_dir("plot_extra_tables"), "tables.md")
        with redirect_stdout(io.StringIO()):
            self.plot.write_tables(self.data, self.summary, path)
        with open(path, encoding="utf-8") as f:
            d_text = f.read().split("## (d)")[1]
        self.assertIn("Configurations with more launches than the others:",
                      d_text)
        self.assertIn("- V2, 8 threads, 100 queries: 4 launches instead of "
                      "3; one extra round was run; see launches.log for the "
                      "reason.", d_text)
        self.assertEqual(d_text.count("one extra round was run"), 1)


# ------------------------------------------------------------ check_results

EF_VALUES = (16, 32, 64, 128, 256)
CHECK_QUERIES = 50
# name, mode, backend, stage_timing, threads of the timed configurations.
CHECK_CONFIGS = [
    ("v1_t1", "cpu", "", "", 1), ("v1_t8", "cpu", "", "", 8),
    ("v2_t1", "hybrid-sync", "cuda", "0", 1),
    ("v2_t8", "hybrid-sync", "cuda", "0", 8),
    ("v2stage_t1", "hybrid-sync", "cuda", "1", 1),
    ("v2stage_t8", "hybrid-sync", "cuda", "1", 8),
    ("v2cpu_t1", "hybrid-sync", "cpu", "0", 1),
    ("v2cpu_t8", "hybrid-sync", "cpu", "0", 8),
]


def check_hash(ef):
    return "%016x" % (0xabc000 + ef)


def check_row(mode, backend, stage, threads, ef, launch, verified):
    row = dict.fromkeys(HEADER, "")
    row.update(build_id=BUILD_ID, host="TESTBOX", build="release",
               build_type="Release", cxx_flags="-O3",
               dataset_hash=DATASET_HASH, queries=CHECK_QUERIES, k=10, ef=ef,
               mode=mode, backend=backend, stage_timing=stage,
               threads=threads, threads_used=threads, launch=launch,
               repetition=1, verified=verified, result_hash=check_hash(ef),
               recall="0.900000", evals_per_query="100.00",
               hops_per_query="10.00", batches_per_query="11.00")
    return row


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


def write_check_fixture(directory):
    """A sweep of one round and one repetition that check_results.py
    accepts; returns the path of the expected-hash table."""
    os.makedirs(directory)
    search = [check_row(mode, backend, stage, threads, ef, 1, "no")
              for _, mode, backend, stage, threads in CHECK_CONFIGS
              for ef in EF_VALUES]
    verify = [check_row("hybrid-sync", "cuda", "0", threads, ef, 0, "yes")
              for threads in (1, 8) for ef in EF_VALUES]
    write_rows(os.path.join(directory, "search.csv"), search)
    write_rows(os.path.join(directory, "verify.csv"), verify)
    expected = os.path.join(directory, "expected.csv")
    with open(expected, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow("dataset_hash queries k ef result_hash recall "
                        "evals_per_query hops_per_query "
                        "batches_per_query".split())
        for ef in EF_VALUES:
            writer.writerow([DATASET_HASH, CHECK_QUERIES, 10, ef,
                             check_hash(ef), "0.900000", "100.00", "10.00",
                             "11.00"])
    return expected


class CheckResultsTests(unittest.TestCase):
    def check(self, edit=None):
        """Runs check_results.py on the fixture after edit(search rows)."""
        directory = os.path.join(scratch_dir("check_" + self._testMethodName),
                                 "sweep")
        expected = write_check_fixture(directory)
        if edit:
            path = os.path.join(directory, "search.csv")
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            rows = edit(rows)
            write_rows(path, rows)
        return run_python("check_results.py", directory, "--expected",
                          expected, "--rounds", 1, "--repetitions", 1,
                          "--build-id", BUILD_ID)

    def test_complete_sweep_passes(self):
        status, out, err = self.check()
        self.assertEqual((status, err), (0, ""), out)
        self.assertTrue(out.startswith("check_results: ok"), out)

    def test_wrong_hash_is_refused(self):
        def edit(rows):
            rows[7]["result_hash"] = "0000000000000000"
            return rows
        status, out, err = self.check(edit)
        self.assertEqual(status, 1, out)
        self.assertIn("result_hash 0000000000000000", err)

    def test_missing_row_is_refused(self):
        status, out, err = self.check(lambda rows: rows[:12] + rows[13:])
        self.assertEqual(status, 1, out)
        self.assertIn("is missing 1 rows", err)


class ExtraLaunchCheckTests(unittest.TestCase):
    def check(self, told, edit=None):
        """Runs check_results.py on the fixture plus launch 2 of v1_t8 and
        v2_t8 (after edit(its rows)), given --extra for each text in told."""
        directory = os.path.join(scratch_dir("check_" + self._testMethodName),
                                 "sweep")
        expected = write_check_fixture(directory)
        extra = [check_row(mode, backend, stage, threads, ef, 2, "no")
                 for name, mode, backend, stage, threads in CHECK_CONFIGS
                 if name in ("v1_t8", "v2_t8") for ef in EF_VALUES]
        if edit:
            extra = edit(extra)
        path = os.path.join(directory, "search.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        write_rows(path, rows + extra)
        options = []
        for text in told:
            options += ["--extra", text]
        return run_python("check_results.py", directory, "--expected",
                          expected, "--rounds", 1, "--repetitions", 1,
                          "--build-id", BUILD_ID, *options)

    def test_extra_launch_passes_when_told(self):
        status, out, err = self.check(["2:v1_t8,v2_t8"])
        self.assertEqual((status, err), (0, ""), out)
        self.assertIn("extra launch 2 for v1_t8 v2_t8", out)

    def test_extra_launch_is_refused_when_not_told(self):
        status, out, err = self.check([])
        self.assertEqual(status, 1, out)
        self.assertIn("unexpected row (mode cpu, backend '', stage_timing "
                      "'', threads 8, queries 50, ef 16, launch 2", err)

    def test_extra_launch_for_other_names_is_refused(self):
        status, out, err = self.check(["2:v1_t8"])
        self.assertEqual(status, 1, out)
        self.assertIn("unexpected row (mode hybrid-sync, backend 'cuda'",
                      err)

    def test_extra_launch_missing_an_ef_is_refused(self):
        status, out, err = self.check(["2:v1_t8,v2_t8"],
                                      lambda rows: rows[:7] + rows[8:])
        self.assertEqual(status, 1, out)
        self.assertIn("v2_t8 round 2 (queries 50) is missing 1 rows, as "
                      "ef/repetition: 64/1", err)

    def test_named_configuration_without_rows_is_refused(self):
        status, out, err = self.check(["2:v1_t8,v2_t8,v2cpu_t1"])
        self.assertEqual(status, 1, out)
        self.assertIn("v2cpu_t1 round 2 (queries 50) is missing 5 rows", err)

    def test_extra_launch_must_follow_the_rounds(self):
        status, out, err = self.check(["1:v1_t8"])
        self.assertEqual(status, 1, out)
        self.assertIn("--extra launch 1 is not above the 1 rounds", err)


# ------------------------------------------------------------- visit_cdf.py

# Ten base vectors and a learn set of ten rows. Visit counts of the runs:
#   first_ef64   F = [9, 0, 5, 1, 3, 2, 8, 10, 4, 6]   total 48
#   second_ef64  S = [2, 6, 0, 3, 9, 0, 1, 7, 4, 8]   total 40
#   all_ef64     F + S = [11, 6, 5, 4, 12, 2, 9, 17, 8, 14]   total 88
# In-sample curve of all_ef64: counts from the largest, 17 14 12 11 9 8 6 5 4
# 2, running totals 17 31 43 54 63 71 77 82 86 88, over 88, for the top 1 to
# 10 vectors.
# Held-out: vectors ranked by F (ids 7 0 6 9 2 8 4 5 3 1, no ties), counted
# with S: S of those ids is 7 2 1 8 0 4 9 0 3 6, running totals 7 9 10 18 18
# 22 31 31 34 40, over 40. The in-sample curve of S has running totals 9 17
# 24 30 34 37 39 40 40 40, over 40, above it at every point.
# Ranking by S and counting F would give 3 9 19 19 23 24 33 41 (46 or 43) 48
# over 48, which is the halves swapped. Ranking S by itself gives the
# in-sample curve of S, which is the held-out curve ranked in-sample.
COUNTS = {
    "all_ef16": [7, 0, 3, 1, 12, 0, 2, 9, 1, 5],
    "all_ef64": [11, 6, 5, 4, 12, 2, 9, 17, 8, 14],
    "all_ef256": [30, 18, 12, 10, 8, 7, 6, 5, 4, 0],
    "first_ef64": [9, 0, 5, 1, 3, 2, 8, 10, 4, 6],
    "second_ef64": [2, 6, 0, 3, 9, 0, 1, 7, 4, 8],
}
EF_OF = {"all_ef16": 16, "all_ef64": 64, "all_ef256": 256, "first_ef64": 64,
         "second_ef64": 64}
RANGE_OF = {"all_ef16": "0,10", "all_ef64": "0,10", "all_ef256": "0,10",
            "first_ef64": "0,5", "second_ef64": "5,5"}
# Running totals of the in-sample curves, by hand as above (ef 16: 12 9 7 5 3
# 2 1 1 0 0; ef 256: 30 18 12 10 8 7 6 5 4 0).
IN_SAMPLE_TOTALS = {
    16: ([12, 21, 28, 33, 36, 38, 39, 40, 40, 40], 40),
    64: ([17, 31, 43, 54, 63, 71, 77, 82, 86, 88], 88),
    256: ([30, 48, 60, 70, 78, 85, 91, 96, 100, 100], 100),
}
HELD_OUT_TOTALS = ([7, 9, 10, 18, 18, 22, 31, 31, 34, 40], 40)


def poisson_share(mean, share):
    """Share of the evaluations held by the most-visited share of vectors
    when counts are Poisson(mean): the vectors with the largest count first,
    the last run of equal counts partly."""
    pmf = [math.exp(-mean)]
    for value in range(1, 200):
        pmf.append(pmf[-1] * mean / value)
    vectors = evals = 0.0
    for value in range(len(pmf) - 1, -1, -1):
        if vectors + pmf[value] >= share:
            return (evals + (share - vectors) * value) / mean
        vectors += pmf[value]
        evals += pmf[value] * value
    return 1.0


def printed_lines(counts):
    """The percentage lines hvs_search prints for a run's counts."""
    ordered = sorted(counts, reverse=True)
    lines = []
    for share in (0.01, 0.05, 0.10, 0.25, 0.50):
        covered = sum(ordered[:int(share * len(ordered))])
        lines.append("  most-visited %4.0f%% of vectors: %5.1f%% of "
                     "evaluations\n" % (share * 100,
                                        100.0 * covered / sum(ordered)))
    return lines


def write_visit_fixture(directory, counts=None, with_lines=True):
    """Count and text files of the five runs; returns the arguments for
    visit_cdf.py."""
    counts = counts or COUNTS
    counts_dir = os.path.join(directory, "counts")
    text_dir = os.path.join(directory, "texts")
    os.makedirs(counts_dir)
    os.makedirs(text_dir)
    for name, values in counts.items():
        with open(os.path.join(counts_dir, name + ".bin"), "wb") as f:
            f.write(struct.pack("<II", len(values), 1))
            f.write(struct.pack("<%dI" % len(values), *values))
        lines = []
        if with_lines:
            lines += ["visit counts over 10 learn queries at ef=%d: %d "
                      "evaluations\n" % (EF_OF[name], sum(values))]
            lines += printed_lines(values)
        lines += ["ef=%d\n" % EF_OF[name], "k=10\n",
                  "query_file=/data/learn.fbin\n",
                  "range=%s\n" % RANGE_OF[name],
                  "evaluations=%d\n" % sum(values),
                  "never_visited=%d\n" % values.count(0), "entry_point=3\n",
                  "dataset_hash=%s\n" % DATASET_HASH,
                  "build_id=%s\n" % BUILD_ID]
        with open(os.path.join(text_dir, name + ".txt"), "w") as f:
            f.writelines(lines)
    out = os.path.join(directory, "visit_cdf.csv")
    return ["--counts-dir", counts_dir, "--text-dir", text_dir,
            "--learn-rows", 10, "--out", out], out


def read_curves(path):
    """{(curve, ef): [(share of vectors, share of evaluations)]}"""
    curves = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            curves.setdefault((row["curve"], int(row["ef"])), []).append(
                (float(row["share_of_vectors"]),
                 float(row["share_of_evaluations"])))
    return curves


class VisitCdfTests(unittest.TestCase):
    def run_tool(self, counts=None, with_lines=True, edit=None):
        directory = scratch_dir("visit_" + self._testMethodName)
        args, out = write_visit_fixture(directory, counts, with_lines)
        if edit:
            edit(directory)
        status, stdout, stderr = run_python("visit_cdf.py", *args)
        return status, stdout, stderr, out

    def assert_curve(self, points, totals, places=9):
        running, total = totals
        self.assertEqual(len(points), 10)
        for j, (share_v, share_e) in enumerate(points, 1):
            self.assertAlmostEqual(share_v, j / 10, places=9)
            self.assertAlmostEqual(share_e, running[j - 1] / total,
                                   places=places, msg="vector %d" % j)

    def test_curves_match_the_hand_computed_ones(self):
        status, out, err, path = self.run_tool()
        self.assertEqual((status, err), (0, ""), out)
        curves = read_curves(path)
        self.assertEqual(sorted(curves), [
            ("held-out", 64), ("in-sample", 16), ("in-sample", 64),
            ("in-sample", 256), ("uniform", 16), ("uniform", 64),
            ("uniform", 256)])
        for ef, totals in IN_SAMPLE_TOTALS.items():
            self.assert_curve(curves[("in-sample", ef)], totals)
        self.assert_curve(curves[("held-out", 64)], HELD_OUT_TOTALS)

    def test_uniform_curve_is_poisson(self):
        status, out, err, path = self.run_tool()
        self.assertEqual(status, 0, err)
        curves = read_curves(path)
        for ef, total in ((16, 40), (64, 88), (256, 100)):
            for share_v, share_e in curves[("uniform", ef)]:
                self.assertAlmostEqual(
                    share_e, poisson_share(total / 10, share_v), places=6)
            # Above y = x everywhere inside, so it is not the diagonal.
            self.assertGreater(curves[("uniform", ef)][4][1], 0.5 + 0.05)
            self.assertLess(curves[("uniform", ef)][0][1], 0.1 + 0.2)

    def test_printed_percentages_are_compared(self):
        status, out, err, _ = self.run_tool()
        self.assertEqual(status, 0, err)
        compared = int(re.search(r"(\d+) printed percentages", out).group(1))
        self.assertEqual(compared, 25)
        # A run without those lines is accepted and nothing is compared.
        status, out, err, _ = self.run_tool(with_lines=False)
        self.assertEqual(status, 0, err)
        self.assertIn("0 printed percentages", out)

        def wrong_line(directory):
            path = os.path.join(directory, "texts", "all_ef64.txt")
            with open(path) as f:
                lines = f.readlines()
            lines[5] = ("  most-visited   50% of vectors:  99.9% of "
                        "evaluations\n")
            with open(path, "w") as f:
                f.writelines(lines)
        status, out, err, _ = self.run_tool(edit=wrong_line)
        self.assertEqual(status, 1, out)
        self.assertIn("all_ef64.txt prints", err)

    def test_flat_counts_are_refused(self):
        # Evenly spread counts are below the uniform baseline.
        flat = dict(COUNTS, all_ef16=[4] * 10)
        status, out, err, _ = self.run_tool(counts=flat, with_lines=False)
        self.assertEqual(status, 1, out)
        self.assertIn("below the uniform", err)


# --------------------------------------------------------------------- main

def plot_modules_missing():
    """Why the plot.py tests cannot run here, or an empty string."""
    for name in ("pandas", "matplotlib"):
        if importlib.util.find_spec(name) is None:
            return "no module named '%s' for %s" % (name, sys.executable)
    return ""


def main():
    global TOOLS, SCRATCH
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scratch", help="scratch directory, emptied first")
    ap.add_argument("--tools", help="directory of plot.py, check_results.py "
                    "and visit_cdf.py (default: $HVS_TOOLS_DIR or ../tools)")
    args = ap.parse_args()
    TOOLS = os.path.abspath(args.tools or os.environ.get("HVS_TOOLS_DIR")
                            or os.path.join(here, "..", "tools"))
    SCRATCH = os.path.abspath(args.scratch)
    shutil.rmtree(SCRATCH, ignore_errors=True)
    os.makedirs(SCRATCH)

    classes = [CheckResultsTests, ExtraLaunchCheckTests, VisitCdfTests]
    skipped = plot_modules_missing()
    if skipped:
        print("test_tools: SKIPPED the plot.py tests: " + skipped)
    else:
        classes[:0] = [PlotTests, ExtraLaunchTests]
    load = unittest.defaultTestLoader.loadTestsFromTestCase
    suite = unittest.TestSuite(load(c) for c in classes)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if skipped:
        print("test_tools: SKIPPED the plot.py tests: " + skipped)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
