/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/io.h"
#include "hvs/types.h"

#include <cstdint>
#include <string>
#include <vector>

// Mean fraction of the true k nearest that each result contains.
double recall_at_k(const std::vector<hvs::SearchResult> &results,
                   const hvs::IdMatrix &gt, uint32_t k);

// Nearest-rank percentile: element ceil(percent * size / 100) - 1 of the
// sorted sample, for percent from 1 to 100. So the 50th percentile of four
// values is the second smallest.
double percentile(std::vector<double> sample, uint32_t percent);

// 64-bit FNV-1a over bytes, continuing from hash.
constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ull;
uint64_t fnv1a(uint64_t hash, const unsigned char *bytes, size_t count);

// The value as 16 lowercase hex digits.
std::string hex64(uint64_t value);

// FNV-1a of the files' bytes, one after the other, as hex.
std::string files_hash(const std::vector<std::string> &paths);

// FNV-1a, as hex, over little-endian uint32 values: the query count, then for
// each query its id, result count, result ids, distance_evals and hops.
// results[i] must be query i. Throws if a value does not fit in 32 bits.
std::string result_hash(const std::vector<hvs::SearchResult> &results);

// The values of one CSV row; README.md describes each column.
struct ReportRow {
  std::string build_id, tag, host, platform, build, compiler;
  std::string build_type, cxx_flags;
  uint32_t logical_cpus = 0;
  std::string omp_proc_bind, omp_places, dataset_hash;
  uint32_t n = 0, d = 0, degree = 0, queries = 0, k = 0, ef = 0;
  std::string mode;
  uint32_t threads = 0, threads_used = 0, launch = 0, repetition = 0;
  uint32_t warmups = 0;
  double recall = 0, qps = 0;
  double latency_mean_us = 0, latency_p50_us = 0, latency_p95_us = 0;
  double latency_p99_us = 0;
  double evals_per_query = 0, hops_per_query = 0, batches_per_query = 0;
  std::string result_hash;
  // hybrid-sync only. Which rows leave which of these columns empty is
  // decided in report.cpp, from mode, backend and stage_timing.
  std::string backend, gpu, cuda_runtime, cuda_driver;
  bool stage_timing = false, verified = false, pinned = false;
  bool resident = false;
  uint32_t max_batch = 0;
  uint64_t device_bytes = 0;
  double init_s = 0, launch_floor_us = 0;
  double round_trips_per_query = 0, candidates_per_round_trip = 0;
  double bytes_to_device_per_query = 0, bytes_from_device_per_query = 0;
  double backend_us = 0, outside_backend_us = 0;
  double gather_us = 0, submit_us = 0, wait_us = 0;
  double to_device_us = 0, kernel_us = 0, from_device_us = 0;
  std::string start_time_utc, command;
};

// The column names, comma-separated, ending in a newline. Every mode writes
// the same columns.
std::string csv_header();

// The row's values in header order, ending in a newline.
std::string csv_line(const ReportRow &row);

// Throws unless path is missing, empty, or starts with header and ends in a
// newline, so rows are never appended under a different header. Also throws
// if path cannot be opened for appending; a missing file is not left behind.
void check_csv(const std::string &path, const std::string &header);

// After check_csv, appends lines to path, with header first if the file is
// missing or empty.
void append_csv(const std::string &path, const std::string &header,
                const std::vector<std::string> &lines);

// What --verify checks one V2 repetition against, besides V1's results.
struct VerifyCounts {
  uint32_t queries = 0;       // queries requested
  uint32_t k = 0, ef = 0;     // results per query and search list size
  uint32_t n = 0, degree = 0; // graph vertices and level-0 degree
  uint32_t max_batch = 0;     // candidates per round trip
  uint64_t round_trips = 0;   // summed over backends; calls, for cpu ones
  uint64_t candidates = 0;    // summed over backends
};

// Compares V2's results of one repetition with V1's, each query in full and
// distances bit for bit, and the backend totals with the results' counters.
// Returns the first failing check, naming the query, or "" if all pass.
std::string compare_results(const std::vector<hvs::SearchResult> &v2,
                            const std::vector<hvs::SearchResult> &v1,
                            const VerifyCounts &c);
