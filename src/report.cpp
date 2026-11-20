/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "report.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace hvs;

double recall_at_k(const std::vector<SearchResult> &results, const IdMatrix &gt,
                   uint32_t k) {
  size_t hits = 0;
  for (const SearchResult &r : results)
    for (VectorId id : r.ids)
      hits += std::count(gt.row(r.query_id), gt.row(r.query_id) + k, id);
  return double(hits) / double(results.size() * k);
}

double percentile(std::vector<double> sample, uint32_t percent) {
  if (sample.empty() || percent == 0 || percent > 100)
    throw std::invalid_argument("percentile needs a sample and 1 to 100");
  std::sort(sample.begin(), sample.end());
  const size_t rank = (size_t(percent) * sample.size() + 99) / 100;
  return sample[rank - 1];
}

uint64_t fnv1a(uint64_t hash, const unsigned char *bytes, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    hash ^= bytes[i];
    hash *= 0x100000001b3ull;
  }
  return hash;
}

std::string hex64(uint64_t value) {
  char text[17];
  std::snprintf(text, sizeof(text), "%016llx", (unsigned long long)value);
  return text;
}

std::string files_hash(const std::vector<std::string> &paths) {
  uint64_t hash = FNV_OFFSET;
  std::vector<char> chunk(1 << 20);
  for (const std::string &path : paths) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
      throw std::runtime_error(path + ": cannot open");
    while (in) {
      in.read(chunk.data(), std::streamsize(chunk.size()));
      hash = fnv1a(hash, reinterpret_cast<const unsigned char *>(chunk.data()),
                   size_t(in.gcount()));
    }
    if (!in.eof())
      throw std::runtime_error(path + ": read error");
  }
  return hex64(hash);
}

// Appends value to bytes as a little-endian uint32.
static void put_u32(std::vector<unsigned char> &bytes, uint64_t value) {
  if (value > UINT32_MAX)
    throw std::runtime_error("result hash: " + std::to_string(value) +
                             " does not fit in 32 bits");
  for (int shift = 0; shift < 32; shift += 8)
    bytes.push_back(static_cast<unsigned char>(value >> shift));
}

std::string result_hash(const std::vector<SearchResult> &results) {
  std::vector<unsigned char> bytes;
  put_u32(bytes, results.size());
  for (size_t q = 0; q < results.size(); ++q) {
    const SearchResult &r = results[q];
    if (r.query_id != q)
      throw std::runtime_error("result hash: results not in query order");
    put_u32(bytes, r.query_id);
    put_u32(bytes, r.ids.size());
    for (VectorId id : r.ids)
      put_u32(bytes, id);
    put_u32(bytes, r.distance_evals);
    put_u32(bytes, r.hops);
  }
  return hex64(fnv1a(FNV_OFFSET, bytes.data(), bytes.size()));
}

// The value with the given number of digits after the point.
static std::string fixed(double value, int digits) {
  char text[64];
  std::snprintf(text, sizeof(text), "%.*f", digits, value);
  return text;
}

static std::string num(uint64_t value) { return std::to_string(value); }

// Which rows fill a column: hybrid-sync rows, rows of the cuda backend, and
// cuda rows without and with --stage-timing. Other rows leave it empty.
static bool hybrid(const ReportRow &r) { return r.mode == "hybrid-sync"; }
static bool cuda(const ReportRow &r) { return r.backend == "cuda"; }
static bool throughput(const ReportRow &r) {
  return cuda(r) && !r.stage_timing;
}
static bool staged(const ReportRow &r) { return cuda(r) && r.stage_timing; }

// The value with two digits after the point if filled, else empty.
static std::string two(bool filled, double value) {
  return filled ? fixed(value, 2) : "";
}

// A CSV column: its name and how a row's value is written.
struct Column {
  const char *name;
  std::string (*value)(const ReportRow &row);
};

// Every fixed column in file order. The header and each row both come from
// here.
static const Column COLUMNS[] = {
    {"build_id", [](const ReportRow &r) { return r.build_id; }},
    {"tag", [](const ReportRow &r) { return r.tag; }},
    {"host", [](const ReportRow &r) { return r.host; }},
    {"platform", [](const ReportRow &r) { return r.platform; }},
    {"build", [](const ReportRow &r) { return r.build; }},
    {"compiler", [](const ReportRow &r) { return r.compiler; }},
    {"build_type", [](const ReportRow &r) { return r.build_type; }},
    {"cxx_flags", [](const ReportRow &r) { return r.cxx_flags; }},
    {"logical_cpus", [](const ReportRow &r) { return num(r.logical_cpus); }},
    {"omp_proc_bind", [](const ReportRow &r) { return r.omp_proc_bind; }},
    {"omp_places", [](const ReportRow &r) { return r.omp_places; }},
    {"dataset_hash", [](const ReportRow &r) { return r.dataset_hash; }},
    {"n", [](const ReportRow &r) { return num(r.n); }},
    {"d", [](const ReportRow &r) { return num(r.d); }},
    {"degree", [](const ReportRow &r) { return num(r.degree); }},
    {"queries", [](const ReportRow &r) { return num(r.queries); }},
    {"k", [](const ReportRow &r) { return num(r.k); }},
    {"ef", [](const ReportRow &r) { return num(r.ef); }},
    {"mode", [](const ReportRow &r) { return r.mode; }},
    {"threads", [](const ReportRow &r) { return num(r.threads); }},
    {"threads_used", [](const ReportRow &r) { return num(r.threads_used); }},
    {"launch", [](const ReportRow &r) { return num(r.launch); }},
    {"repetition", [](const ReportRow &r) { return num(r.repetition); }},
    {"warmups", [](const ReportRow &r) { return num(r.warmups); }},
    {"recall", [](const ReportRow &r) { return fixed(r.recall, 6); }},
    {"qps", [](const ReportRow &r) { return fixed(r.qps, 1); }},
    {"latency_mean_us",
     [](const ReportRow &r) { return fixed(r.latency_mean_us, 2); }},
    {"latency_p50_us",
     [](const ReportRow &r) { return fixed(r.latency_p50_us, 2); }},
    {"latency_p95_us",
     [](const ReportRow &r) { return fixed(r.latency_p95_us, 2); }},
    {"latency_p99_us",
     [](const ReportRow &r) { return fixed(r.latency_p99_us, 2); }},
    {"evals_per_query",
     [](const ReportRow &r) { return fixed(r.evals_per_query, 2); }},
    {"hops_per_query",
     [](const ReportRow &r) { return fixed(r.hops_per_query, 2); }},
    {"batches_per_query",
     [](const ReportRow &r) { return fixed(r.batches_per_query, 2); }},
    {"result_hash", [](const ReportRow &r) { return r.result_hash; }},
    {"backend", [](const ReportRow &r) { return r.backend; }},
    {"stage_timing",
     [](const ReportRow &r) { return hybrid(r) ? num(r.stage_timing) : ""; }},
    {"verified",
     [](const ReportRow &r) { return std::string(r.verified ? "yes" : "no"); }},
    {"gpu", [](const ReportRow &r) { return r.gpu; }},
    {"cuda_runtime", [](const ReportRow &r) { return r.cuda_runtime; }},
    {"cuda_driver", [](const ReportRow &r) { return r.cuda_driver; }},
    {"wait_policy",
     [](const ReportRow &r) { return std::string(cuda(r) ? "auto" : ""); }},
    {"max_batch",
     [](const ReportRow &r) { return cuda(r) ? num(r.max_batch) : ""; }},
    {"pinned", [](const ReportRow &r) { return cuda(r) ? num(r.pinned) : ""; }},
    {"resident",
     [](const ReportRow &r) { return cuda(r) ? num(r.resident) : ""; }},
    {"init_s",
     [](const ReportRow &r) { return hybrid(r) ? fixed(r.init_s, 4) : ""; }},
    {"device_bytes",
     [](const ReportRow &r) { return cuda(r) ? num(r.device_bytes) : ""; }},
    {"launch_floor_us",
     [](const ReportRow &r) { return two(cuda(r), r.launch_floor_us); }},
    {"round_trips_per_query",
     [](const ReportRow &r) {
       return two(hybrid(r), r.round_trips_per_query);
     }},
    {"candidates_per_round_trip",
     [](const ReportRow &r) {
       return two(hybrid(r), r.candidates_per_round_trip);
     }},
    {"bytes_to_device_per_query",
     [](const ReportRow &r) {
       return two(cuda(r), r.bytes_to_device_per_query);
     }},
    {"bytes_from_device_per_query",
     [](const ReportRow &r) {
       return two(cuda(r), r.bytes_from_device_per_query);
     }},
    {"backend_us",
     [](const ReportRow &r) { return two(cuda(r), r.backend_us); }},
    {"outside_backend_us",
     [](const ReportRow &r) { return two(cuda(r), r.outside_backend_us); }},
    {"gather_us", [](const ReportRow &r) { return two(cuda(r), r.gather_us); }},
    {"submit_us",
     [](const ReportRow &r) { return two(throughput(r), r.submit_us); }},
    {"wait_us",
     [](const ReportRow &r) { return two(throughput(r), r.wait_us); }},
    {"to_device_us",
     [](const ReportRow &r) { return two(staged(r), r.to_device_us); }},
    {"kernel_us",
     [](const ReportRow &r) { return two(staged(r), r.kernel_us); }},
    {"from_device_us",
     [](const ReportRow &r) { return two(staged(r), r.from_device_us); }},
    {"start_time_utc", [](const ReportRow &r) { return r.start_time_utc; }},
    {"command", [](const ReportRow &r) { return r.command; }},
};

// Each field is followed by a comma; the last comma becomes the newline.
std::string csv_header() {
  std::string line;
  for (const Column &c : COLUMNS)
    line += std::string(c.name) + ",";
  line.back() = '\n';
  return line;
}

std::string csv_line(const ReportRow &row) {
  std::string line;
  for (const Column &c : COLUMNS)
    line += c.value(row) + ",";
  line.back() = '\n';
  return line;
}

void check_csv(const std::string &path, const std::string &header) {
  std::ifstream in(path, std::ios::binary);
  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  if (!text.empty() && text.compare(0, header.size(), header) != 0)
    throw std::runtime_error(path + ": first line is not this program's CSV "
                                    "header; not appending to it");
  if (!text.empty() && text.back() != '\n')
    throw std::runtime_error(path + ": last line is incomplete; not "
                                    "appending to it");
  // Opening for append creates a missing file, which is removed again.
  std::error_code error;
  const bool existed = std::filesystem::exists(path, error);
  const bool opened =
      bool(std::ofstream(path, std::ios::binary | std::ios::app));
  if (!existed)
    std::filesystem::remove(path, error);
  if (!opened)
    throw std::runtime_error(path + ": cannot open for appending");
}

void append_csv(const std::string &path, const std::string &header,
                const std::vector<std::string> &lines) {
  check_csv(path, header);
  const bool empty =
      std::ifstream(path, std::ios::binary | std::ios::ate).tellg() <= 0;
  std::ofstream out(path, std::ios::binary | std::ios::app);
  if (empty)
    out << header;
  for (const std::string &line : lines)
    out << line;
  out.close();
  if (!out)
    throw std::runtime_error("cannot write " + path);
}

std::string compare_results(const std::vector<SearchResult> &v2,
                            const std::vector<SearchResult> &v1,
                            const VerifyCounts &c) {
  if (c.queries == 0 || v2.size() != c.queries || v1.size() != c.queries)
    return "compared " + num(v2.size()) + " V2 and " + num(v1.size()) +
           " V1 queries, expected " + num(c.queries);
  if (c.max_batch < c.degree)
    return "max_batch " + num(c.max_batch) + " is below the degree " +
           num(c.degree) + ", so a batch could take two round trips";
  // Entries per result list: min(k, ef), unless the graph is smaller.
  const size_t full = std::min({c.k, c.ef, c.n});
  uint64_t evals = 0, batches = 0;
  for (size_t q = 0; q < v2.size(); ++q) {
    const SearchResult &x = v2[q], &y = v1[q];
    const std::string at = "query " + num(q) + ": ";
    if (x.query_id != q || y.query_id != q)
      return at + "results are not in query order";
    if (x.ids.size() != full || x.distances.size() != full ||
        y.ids.size() != full || y.distances.size() != full)
      return at + "a result list does not have " + num(full) + " entries";
    if (x.ids != y.ids)
      return at + "ids differ";
    for (size_t i = 0; i < full; ++i)
      if (std::memcmp(&x.distances[i], &y.distances[i], sizeof(float)) != 0)
        return at + "distance " + num(i) + " differs";
    if (x.distance_evals != y.distance_evals)
      return at + "distance_evals differ";
    if (x.hops != y.hops)
      return at + "hops differ";
    if (x.batches != y.batches)
      return at + "batches differ";
    evals += x.distance_evals;
    batches += x.batches;
  }
  if (c.candidates != evals)
    return "the backends received " + num(c.candidates) +
           " candidates but the results count " + num(evals) + " evaluations";
  if (c.round_trips != batches)
    return "the backends made " + num(c.round_trips) +
           " round trips but the results count " + num(batches) + " batches";
  return "";
}
