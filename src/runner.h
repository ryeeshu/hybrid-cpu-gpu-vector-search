/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "options.h"
#include "report.h"

#include "hvs/backend.h"
#include "hvs/io.h"
#include "hvs/types.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The files of one data directory, checked to agree on shape.
struct Dataset {
  hvs::FloatMatrix base;
  hvs::FloatMatrix queries; // only the first --query-count rows, if given
  hvs::IdMatrix gt;
  hvs::Graph graph;
  // FNV-1a of base.fbin, query.fbin, graph0.bin, upper.bin and gt.ibin
  std::string hash;
};

// The id of the sources and build settings of this binary.
std::string build_id();

// The text with commas, quotes and line breaks replaced, so it is one field.
std::string csv_safe(std::string text);

// Throws if this build cannot run o, or o.csv cannot take its rows, before
// any data is loaded.
void check_build(const Options &o);

Dataset load_dataset(const Options &o);

// Statistics of one pass, summed over the backends of a hybrid-sync launch.
// The cpu backend fills only round_trips, with its calls, and candidates.
struct BackendTotals {
  uint64_t round_trips = 0, candidates = 0;
  uint64_t bytes_to_device = 0, bytes_from_device = 0;
  double round_trip_us = 0, gather_us = 0, submit_us = 0, wait_us = 0;
  double to_device_us = 0, kernel_us = 0, from_device_us = 0;
};

// --verify at one ef: runs V1 with config, then checks every timed V2
// repetition's result hash against V1's (hashes) and the last repetition in
// full with compare_results (last, and totals of its backends). Returns the
// first failing check, or "" if all pass.
std::string verify_against_v1(const Dataset &data,
                              const hvs::SearchConfig &config,
                              uint32_t max_batch,
                              const std::vector<std::string> &hashes,
                              const std::vector<hvs::SearchResult> &last,
                              const BackendTotals &totals);

// Counts how often each base vector's distance is computed, with the learn
// vectors of --visit-range as queries, and writes the counts to
// o.visit_counts. Prints a summary, then key=value lines. Never timed.
void write_visit_counts(const Options &o, const Dataset &data);

// What one pass over the queries produced. totals stays zero in cpu mode.
struct Pass {
  std::vector<hvs::SearchResult> results;
  std::vector<double> latencies_us; // per query, microseconds
  uint32_t threads_used = 0;
  double seconds = 0; // wall time of the whole pass
  BackendTotals totals;
};

// Sets the columns of row that come from one timed pass.
void describe_pass(ReportRow &row, const Pass &pass, const Dataset &data);

// The columns that are the same for every row of this launch, with the
// start time of the launch.
ReportRow launch_row(const Options &o, const Dataset &data);

// The backends of a hybrid-sync launch, one per thread: thread t uses
// slots[t]. Empty in cpu mode.
struct LaunchBackends {
  std::vector<hvs::DistanceBackend *> slots;
  virtual ~LaunchBackends() = default;
  // Clears every backend's statistics, so the next read covers one pass.
  virtual void reset_stats() = 0;
  // Every backend's statistics since reset_stats, summed.
  virtual BackendTotals read_stats() const = 0;
};

// Builds the backends o asks for, serially, and gives each the queries. Sets
// the columns of row that describe them.
std::unique_ptr<LaunchBackends>
build_backends(const Options &o, const Dataset &data, ReportRow &row);

// Times the search at each ef on backends (hybrid-sync) or the CPU. Prints
// one line per ef, checks --verify and, with --csv, appends one row per
// repetition once every ef has run. row holds the launch columns.
void run_benchmark(const Options &o, const Dataset &data, ReportRow row,
                   LaunchBackends &backends);
