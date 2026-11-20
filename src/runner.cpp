/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "runner.h"
#include "build_id.h" // generated at build time by tools/build_id.cmake
#include "report.h"

#include "hvs/cpu_backend.h"
#include "hvs/search.h"

#ifdef HVS_CUDA
#include "hvs/gpu/cuda_backend.h"
constexpr bool HAS_CUDA = true;
#else
constexpr bool HAS_CUDA = false;
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

using namespace hvs;

// Passes per ef that run before the timed repetitions and are discarded.
constexpr uint32_t WARMUPS = 1;

std::string build_id() { return HVS_BUILD_ID; }

void check_build(const Options &o) {
  if (o.backend == "cuda" && !HAS_CUDA)
    throw std::runtime_error("--backend cuda needs a build with HVS_CUDA on "
                             "(preset wsl or ghc); use --backend cpu here");
  if (!o.csv.empty())
    check_csv(o.csv, csv_header());
}

Dataset load_dataset(const Options &o) {
  Dataset data;
  data.base = load_matrix<float>(o.data + "/base.fbin");
  data.queries = load_matrix<float>(o.data + "/query.fbin");
  data.gt = load_matrix<VectorId>(o.data + "/gt.ibin");
  data.graph = load_graph(o.data + "/graph0.bin", o.data + "/upper.bin");
  if (data.queries.d != data.base.d || data.graph.n != data.base.n ||
      data.gt.n != data.queries.n || data.gt.d < o.k)
    throw std::runtime_error("dataset files disagree on shape");
  if (o.query_count != 0 && o.query_count < data.queries.n) {
    data.queries.n = o.query_count;
    data.queries.data.resize(size_t(data.queries.n) * data.queries.d);
  }
  data.hash = files_hash({o.data + "/base.fbin", o.data + "/query.fbin",
                          o.data + "/graph0.bin", o.data + "/upper.bin",
                          o.data + "/gt.ibin"});
  return data;
}

// Per-thread arrays, merged at the end.
void write_visit_counts(const Options &o, const Dataset &data) {
  const Graph &graph = data.graph;
  const FloatMatrix &base = data.base;
  const FloatMatrix learn = load_matrix<float>(o.data + "/learn.fbin");
  if (learn.d != base.d)
    throw std::runtime_error("learn.fbin dimension differs from the base");
  const uint32_t first = o.visit_first;
  const uint32_t count = o.visit_count ? o.visit_count : learn.n - first;
  if (first >= learn.n || uint64_t(first) + count > learn.n)
    throw std::runtime_error("--visit-range is outside the " +
                             std::to_string(learn.n) + " rows of learn.fbin");
  std::vector<uint32_t> counts(base.n, 0);
#pragma omp parallel num_threads(o.threads)
  {
    std::vector<uint32_t> local(base.n, 0);
    QuerySearch search(graph, o.ef[0], o.k);
    std::vector<VectorId> ids;
    std::vector<float> distances;
#pragma omp for schedule(dynamic, 64)
    for (int64_t q = first; q < int64_t(first) + count; ++q) {
      search.begin();
      while (search.next(ids)) {
        distances.resize(ids.size());
        for (size_t i = 0; i < ids.size(); ++i) {
          ++local[ids[i]];
          distances[i] = l2(learn.row(q), base.row(ids[i]), base.d);
        }
        search.absorb(ids, distances.data());
      }
    }
#pragma omp critical
    for (uint32_t i = 0; i < base.n; ++i)
      counts[i] += local[i];
  }

  std::ofstream out(o.visit_counts, std::ios::binary);
  const uint32_t header[2] = {base.n, 1};
  out.write(reinterpret_cast<const char *>(header), sizeof(header));
  out.write(reinterpret_cast<const char *>(counts.data()),
            std::streamsize(counts.size() * sizeof(uint32_t)));
  if (!out)
    throw std::runtime_error("cannot write " + o.visit_counts);

  std::sort(counts.begin(), counts.end(), std::greater<>());
  const uint64_t total =
      std::accumulate(counts.begin(), counts.end(), uint64_t(0));
  const size_t never = size_t(std::count(counts.begin(), counts.end(), 0u));
  std::printf("visit counts over %u learn queries at ef=%u: %llu evaluations\n",
              count, o.ef[0], (unsigned long long)total);
  uint64_t covered = 0;
  size_t i = 0;
  for (double share : {0.01, 0.05, 0.10, 0.25, 0.50}) {
    for (; i < size_t(share * base.n); ++i)
      covered += counts[i];
    std::printf("  most-visited %4.0f%% of vectors: %5.1f%% of evaluations\n",
                share * 100, 100.0 * double(covered) / double(total));
  }
  // The same facts as key=value lines, for scripts.
  std::printf("ef=%u\nk=%u\nquery_file=%s/learn.fbin\nrange=%u,%u\n", o.ef[0],
              o.k, o.data.c_str(), first, count);
  std::printf("evaluations=%llu\nnever_visited=%zu\nentry_point=%u\n",
              (unsigned long long)total, never, graph.entry_point);
  std::printf("dataset_hash=%s\nbuild_id=%s\n", data.hash.c_str(),
              HVS_BUILD_ID);
}

static double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

// One pass of V1, the CPU-only search. Only the search itself is timed.
static Pass run_cpu_pass(const Dataset &data, const SearchConfig &config) {
  Pass pass;
  const auto start = std::chrono::steady_clock::now();
  pass.results = search_all_cpu(data.graph, data.base, data.queries, config,
                                &pass.latencies_us, &pass.threads_used);
  pass.seconds = seconds_since(start);
  return pass;
}

// "wsl2" if /proc/version names Microsoft's kernel, else "linux".
static std::string platform() {
  std::ifstream in("/proc/version");
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  for (char &c : text)
    c = char(std::tolower(static_cast<unsigned char>(c)));
  return text.find("microsoft") != text.npos ? "wsl2" : "linux";
}

// Compiler name and version, such as gcc-13.3.0.
static std::string compiler() {
#if defined(__clang__)
  return "clang-" + std::to_string(__clang_major__) + "." +
         std::to_string(__clang_minor__) + "." +
         std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
  return "gcc-" + std::to_string(__GNUC__) + "." +
         std::to_string(__GNUC_MINOR__) + "." +
         std::to_string(__GNUC_PATCHLEVEL__);
#else
  return "unknown";
#endif
}

std::string csv_safe(std::string text) {
  std::replace(text.begin(), text.end(), ',', ';');
  for (char bad : {'"', '\'', '\n', '\r'})
    std::replace(text.begin(), text.end(), bad, '_');
  return text;
}

// An environment variable as a CSV field, or "unset" if it is unset or empty.
static std::string env_field(const char *name) {
  const char *value = std::getenv(name);
  return value && *value ? csv_safe(value) : "unset";
}

// The backends this program builds. Only the vector of the chosen backend is
// filled.
struct Backends : LaunchBackends {
  std::vector<std::unique_ptr<CpuDistanceBackend>> cpus;
#ifdef HVS_CUDA
  std::vector<std::unique_ptr<CudaDistanceBackend>> gpus;
#endif
  void reset_stats() override;
  BackendTotals read_stats() const override;
};

std::unique_ptr<LaunchBackends>
build_backends(const Options &o, const Dataset &data, ReportRow &row) {
  auto backends = std::make_unique<Backends>();
  Backends &b = *backends;
  if (o.mode != "hybrid-sync")
    return backends;
  const FloatMatrix &queries = data.queries;
  const auto start = std::chrono::steady_clock::now();
  if (o.backend == "cpu") {
    for (uint32_t t = 0; t < o.threads; ++t) {
      b.cpus.push_back(std::make_unique<CpuDistanceBackend>(data.base));
      b.cpus.back()->set_queries(queries.data.data(), queries.n, queries.d);
      b.slots.push_back(b.cpus.back().get());
    }
    row.init_s = seconds_since(start);
  }
#ifdef HVS_CUDA
  if (o.backend == "cuda") {
    CudaBackendOptions options;
    options.max_batch = data.graph.degree; // one round trip per search batch
    options.pinned = true;
    options.resident = false;
    options.stage_timing = o.stage_timing;
    for (uint32_t t = 0; t < o.threads; ++t) {
      b.gpus.push_back(
          std::make_unique<CudaDistanceBackend>(data.base, options));
      b.gpus.back()->set_queries(queries.data.data(), queries.n, queries.d);
      b.slots.push_back(b.gpus.back().get());
    }
    row.init_s = seconds_since(start);
    for (const auto &gpu : b.gpus)
      row.device_bytes += gpu->device_bytes();
    row.max_batch = options.max_batch;
    row.pinned = options.pinned;
    row.resident = options.resident;
    row.launch_floor_us = b.gpus[0]->launch_floor_us(2000); // empty kernels
    const CudaDeviceInfo device = cuda_device_info();
    row.gpu = csv_safe(device.name);
    row.cuda_runtime = device.runtime;
    row.cuda_driver = device.driver;
  }
#endif
  if (b.slots.size() != o.threads)
    throw std::runtime_error("no " + o.backend + " backend in this build");
  return backends;
}

void Backends::reset_stats() {
  for (const auto &cpu : cpus)
    cpu->reset_counts();
#ifdef HVS_CUDA
  for (const auto &gpu : gpus)
    gpu->reset_stats();
#endif
}

BackendTotals Backends::read_stats() const {
  BackendTotals t;
  for (const auto &cpu : cpus) {
    t.round_trips += cpu->calls();
    t.candidates += cpu->candidates();
  }
#ifdef HVS_CUDA
  for (const auto &gpu : gpus) {
    const CudaBackendStats &s = gpu->stats();
    t.round_trips += s.round_trips;
    t.candidates += s.candidates;
    t.bytes_to_device += s.bytes_to_device;
    t.bytes_from_device += s.bytes_from_device;
    t.round_trip_us += s.round_trip_us;
    t.gather_us += s.gather_us;
    t.submit_us += s.submit_us;
    t.wait_us += s.wait_us;
    t.to_device_us += s.to_device_us;
    t.kernel_us += s.kernel_us;
    t.from_device_us += s.from_device_us;
  }
#endif
  return t;
}

// One pass of V2, the synchronous hybrid, with the backends' statistics of
// this pass alone. Only the search itself is timed.
static Pass run_hybrid_pass(const Dataset &data, const SearchConfig &config,
                            LaunchBackends &backends) {
  backends.reset_stats();
  Pass pass;
  const auto start = std::chrono::steady_clock::now();
  pass.results =
      search_all_hybrid_sync(data.graph, data.queries, config, backends.slots,
                             &pass.latencies_us, &pass.threads_used);
  pass.seconds = seconds_since(start);
  pass.totals = backends.read_stats();
  return pass;
}

std::string verify_against_v1(const Dataset &data, const SearchConfig &config,
                              uint32_t max_batch,
                              const std::vector<std::string> &hashes,
                              const std::vector<SearchResult> &last,
                              const BackendTotals &totals) {
  const std::vector<SearchResult> v1 =
      search_all_cpu(data.graph, data.base, data.queries, config);
  const std::string v1_hash = result_hash(v1);
  if (hashes.empty())
    return "no repetition to compare";
  for (size_t r = 0; r < hashes.size(); ++r)
    if (hashes[r] != v1_hash)
      return "repetition " + std::to_string(r + 1) + ": result_hash " +
             hashes[r] + " differs from V1's " + v1_hash;
  VerifyCounts c;
  c.queries = data.queries.n;
  c.k = config.k;
  c.ef = config.ef_search;
  c.n = data.graph.n;
  c.degree = data.graph.degree;
  c.max_batch = max_batch;
  c.round_trips = totals.round_trips;
  c.candidates = totals.candidates;
  return compare_results(last, v1, c);
}

// The current time in UTC, such as 2026-11-19T12:34:56Z.
static std::string utc_now() {
  const std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char text[32];
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return text;
}

ReportRow launch_row(const Options &o, const Dataset &data) {
  char host[256] = "unknown";
  gethostname(host, sizeof(host) - 1);
  ReportRow row;
  row.start_time_utc = utc_now();
  row.command = csv_safe(o.command);
  row.build_id = HVS_BUILD_ID;
  row.build_type = HVS_BUILD_TYPE;
  row.cxx_flags = csv_safe(HVS_CXX_FLAGS);
  row.tag = o.tag;
  row.host = csv_safe(host);
  row.platform = platform();
#ifdef NDEBUG
  row.build = "release";
#else
  row.build = "debug";
#endif
  row.compiler = compiler();
  row.logical_cpus = std::thread::hardware_concurrency();
  row.omp_proc_bind = env_field("OMP_PROC_BIND");
  row.omp_places = env_field("OMP_PLACES");
  row.dataset_hash = data.hash;
  row.n = data.base.n;
  row.d = data.base.d;
  row.degree = data.graph.degree;
  row.queries = data.queries.n;
  row.k = o.k;
  row.mode = o.mode;
  row.threads = o.threads;
  row.launch = o.launch;
  row.warmups = WARMUPS;
  row.backend = o.backend;
  row.stage_timing = o.stage_timing;
  return row;
}

void describe_pass(ReportRow &row, const Pass &pass, const Dataset &data) {
  uint64_t evals = 0, hops = 0, batches = 0;
  for (const SearchResult &r : pass.results) {
    evals += r.distance_evals;
    hops += r.hops;
    batches += r.batches;
  }
  const double queries = double(pass.results.size());
  const std::vector<double> &latencies = pass.latencies_us;
  const double latency_sum =
      std::accumulate(latencies.begin(), latencies.end(), 0.0);
  row.threads_used = pass.threads_used;
  row.recall = recall_at_k(pass.results, data.gt, row.k);
  row.qps = queries / pass.seconds;
  row.latency_mean_us = latency_sum / double(latencies.size());
  row.latency_p50_us = percentile(latencies, 50);
  row.latency_p95_us = percentile(latencies, 95);
  row.latency_p99_us = percentile(latencies, 99);
  row.evals_per_query = double(evals) / queries;
  row.hops_per_query = double(hops) / queries;
  row.batches_per_query = double(batches) / queries;
  row.result_hash = result_hash(pass.results);

  // Backend columns, per query; report.cpp leaves those a row lacks empty.
  const BackendTotals &t = pass.totals;
  row.round_trips_per_query = double(t.round_trips) / queries;
  row.candidates_per_round_trip =
      t.round_trips ? double(t.candidates) / double(t.round_trips) : 0;
  row.bytes_to_device_per_query = double(t.bytes_to_device) / queries;
  row.bytes_from_device_per_query = double(t.bytes_from_device) / queries;
  row.backend_us = t.round_trip_us / queries;
  // Thread time outside the backend calls, glue around each call included.
  row.outside_backend_us = (latency_sum - t.round_trip_us) / queries;
  row.gather_us = t.gather_us / queries;
  row.submit_us = t.submit_us / queries;
  row.wait_us = t.wait_us / queries;
  row.to_device_us = t.to_device_us / queries;
  row.kernel_us = t.kernel_us / queries;
  row.from_device_us = t.from_device_us / queries;
}

void run_benchmark(const Options &o, const Dataset &data, ReportRow row,
                   LaunchBackends &backends) {
  // Checked before anything runs, and again when the rows are appended.
  const std::string header = csv_header();
  if (!o.csv.empty())
    check_csv(o.csv, header);

  std::printf("%s  n=%u d=%u queries=%u k=%u threads=%u (%s build, entry "
              "point fixed, level 0 only)\n",
              row.host.c_str(), row.n, row.d, row.queries, o.k, o.threads,
              row.build.c_str());
  // The backends were built once per launch, before this call.
  if (o.mode == "hybrid-sync")
    std::printf("hybrid-sync, %s backend: built in %.3f s\n", o.backend.c_str(),
                row.init_s);
  // The cpu backend never splits a batch.
  const uint32_t max_batch = o.backend == "cuda" ? row.max_batch : UINT32_MAX;
  const auto run_pass = [&](const SearchConfig &config) {
    return o.mode == "cpu" ? run_cpu_pass(data, config)
                           : run_hybrid_pass(data, config, backends);
  };

  std::printf("%6s %8s %10s %12s %12s %10s %8s\n", "ef", "recall", "QPS",
              "mean us", "p99 us", "evals/q", "hops/q");
  std::vector<ReportRow> rows;
  uint32_t verified_efs = 0; // efs at which every --verify check passed
  for (uint32_t ef : o.ef) {
    SearchConfig config;
    config.k = o.k;
    config.ef_search = ef;
    config.cpu_threads = o.threads;

    for (uint32_t w = 0; w < WARMUPS; ++w)
      run_pass(config); // discarded
    std::vector<Pass> passes;
    for (uint32_t r = 0; r < o.repetitions; ++r)
      passes.push_back(run_pass(config));

    const std::string at = "ef=" + std::to_string(ef) + ": ";
    std::vector<ReportRow> reps;
    std::vector<std::string> hashes;
    for (const Pass &pass : passes) {
      row.ef = ef;
      row.repetition = uint32_t(reps.size()) + 1;
      describe_pass(row, pass, data);
      if (row.threads_used != row.threads)
        throw std::runtime_error(at + "asked for " +
                                 std::to_string(row.threads) +
                                 " threads but the search ran on " +
                                 std::to_string(row.threads_used) +
                                 "; check OMP_THREAD_LIMIT and OMP_DYNAMIC");
      if (!reps.empty() && row.result_hash != reps[0].result_hash)
        throw std::runtime_error(at + "repetitions gave different results (" +
                                 reps[0].result_hash + ", " + row.result_hash +
                                 ")");
      reps.push_back(row);
      rows.push_back(row);
      hashes.push_back(row.result_hash);
    }
    if (o.verify) {
      const std::string failure =
          verify_against_v1(data, config, max_batch, hashes,
                            passes.back().results, passes.back().totals);
      if (!failure.empty())
        throw std::runtime_error(at + "--verify failed: " + failure);
      ++verified_efs;
    }

    // The console shows the repetition with the median QPS.
    std::sort(
        reps.begin(), reps.end(),
        [](const ReportRow &a, const ReportRow &b) { return a.qps < b.qps; });
    const ReportRow &median = reps[reps.size() / 2];
    std::printf("%6u %8.4f %10.0f %12.1f %12.1f %10.1f %8.1f\n", ef,
                median.recall, median.qps, median.latency_mean_us,
                median.latency_p99_us, median.evals_per_query,
                median.hops_per_query);
  }
  // A failed check has thrown by now, so nothing is written.
  std::vector<std::string> lines;
  for (ReportRow &r : rows) {
    r.verified = verified_efs == o.ef.size();
    lines.push_back(csv_line(r));
  }
  if (o.verify)
    std::printf("--verify passed: identical to V1 at every ef\n");
  if (!o.csv.empty())
    append_csv(o.csv, header, lines);
}
