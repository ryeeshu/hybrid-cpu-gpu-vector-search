/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "options.h"
#include "report.h"
#include "runner.h"

#include "hvs/cpu_backend.h"
#include "hvs/search.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace hvs;

// True if calling f throws.
template <typename F> static bool throws(F &&f) {
  try {
    f();
  } catch (const std::exception &) {
    return true;
  }
  return false;
}

// Parses the arguments as hvs_search would.
static Options parse_args(std::vector<std::string> args) {
  args.insert(args.begin(), "hvs_search");
  std::vector<char *> argv;
  for (std::string &arg : args)
    argv.push_back(&arg[0]);
  return parse(int(argv.size()), argv.data());
}

// True if --data d plus flag and value is accepted.
static bool accepts(const std::string &flag, const std::string &value) {
  return !throws([&] { parse_args({"--data", "d", flag, value}); });
}

static void test_options() {
  const Options o =
      parse_args({"--data", "d", "--k", "5", "--threads", "8", "--ef", "16,064",
                  "--repetitions", "2", "--launch", "3", "--tag", "run-a_1.x"});
  CHECK(o.data == "d" && o.k == 5 && o.threads == 8 && o.repetitions == 2);
  CHECK(o.ef.size() == 2 && o.ef[0] == 16 && o.ef[1] == 64);
  CHECK(o.launch == 3 && o.tag == "run-a_1.x");
  CHECK(parse_args({"--data", "d"}).launch == 1);
  CHECK(parse_u32("--x", "4294967295") == 4294967295u);
  CHECK(parse_u32("--x", "0") == 0);
  CHECK(accepts("--launch", "0") && accepts("--query-count", "0"));

  for (const char *bad : {"-1", "+1", "4x", "x4", "", " 4", "4 ", "1.5", "0x10",
                          "1e3", "4294967296", "18446744073709551617"})
    CHECK(!accepts("--threads", bad));
  for (const char *flag : {"--k", "--query-count", "--repetitions", "--launch"})
    CHECK(!accepts(flag, "-1") && !accepts(flag, "4x") && !accepts(flag, ""));
  for (const char *bad : {"4294967297", "16,,64", "16,", ",16", "", "0", "16,0",
                          "-16", "16 ,64", "16,16"})
    CHECK(!accepts("--ef", bad));
  for (const char *flag : {"--k", "--threads", "--repetitions"})
    CHECK(!accepts(flag, "0"));
  for (const char *bad : {"a,b", "a\"b", "a'b", "a\nb", "a\rb", ""})
    CHECK(!accepts("--tag", bad));
  CHECK(!accepts("--mode", "gpu"));
  CHECK(!accepts("--bogus", "1"));
  CHECK(throws([] { parse_args({"--data"}); }));
  CHECK(throws([] { parse_args({"--k", "5"}); }));

  // Modes, backends and the flags without a value.
  const Options cpu = parse_args({"--data", "d"});
  CHECK(cpu.mode == "cpu" && cpu.backend.empty());
  CHECK(!cpu.verify && !cpu.stage_timing);
  const Options gpu = parse_args({"--data", "d", "--mode", "hybrid-sync",
                                  "--stage-timing", "--verify", "--k", "3"});
  CHECK(gpu.backend == "cuda" && gpu.stage_timing && gpu.verify && gpu.k == 3);
  const Options cpus = parse_args(
      {"--verify", "--data", "d", "--backend", "cpu", "--mode", "hybrid-sync"});
  CHECK(cpus.backend == "cpu" && cpus.verify && !cpus.stage_timing);
  for (std::vector<std::string> bad : std::vector<std::vector<std::string>>{
           {"--mode", "hybrid"},
           {"--mode", "hybrid-sync", "--backend", "gpu"},
           {"--backend", "cpu"},
           {"--mode", "cpu", "--backend", "cuda"},
           {"--verify"},
           {"--mode", "cpu", "--verify"},
           {"--stage-timing"},
           {"--mode", "hybrid-sync", "--backend", "cpu", "--stage-timing"},
           {"--mode", "hybrid-sync", "--verify", "1"},
           {"--visit-range", "1,2"},
           {"--visit-counts", "f", "--visit-range", "5"},
           {"--visit-counts", "f", "--visit-range", "1,0"},
           {"--visit-counts", "f", "--visit-range", ",5"},
           {"--visit-counts", "f", "--visit-range", "5,"},
           {"--visit-counts", "f", "--visit-range", "1,2,3"}}) {
    bad.insert(bad.begin(), {"--data", "d"});
    CHECK(throws([&] { parse_args(bad); }));
  }

  // --build-id needs nothing else; the command line is kept for the CSV.
  CHECK(parse_args({"--build-id"}).print_build_id);
  CHECK(parse_args({"--data", "d", "--k", "5"}).command ==
        "hvs_search --data d --k 5");
  const Options v = parse_args(
      {"--data", "d", "--visit-counts", "f", "--visit-range", "0,7"});
  CHECK(v.visit_first == 0 && v.visit_count == 7);
}

static void test_percentile() {
  const std::vector<double> four = {40, 10, 30, 20};
  CHECK(percentile(four, 1) == 10);
  CHECK(percentile(four, 25) == 10);
  CHECK(percentile(four, 26) == 20);
  CHECK(percentile(four, 50) == 20); // the lower median
  CHECK(percentile(four, 51) == 30);
  CHECK(percentile(four, 95) == 40);
  CHECK(percentile(four, 100) == 40);
  CHECK(percentile({7}, 1) == 7 && percentile({7}, 99) == 7);
  std::vector<double> sample;
  for (int i = 1000; i >= 1; --i)
    sample.push_back(i);
  CHECK(percentile(sample, 50) == 500);
  CHECK(percentile(sample, 95) == 950);
  CHECK(percentile(sample, 99) == 990);
  CHECK(throws([] { percentile({}, 50); }));
  CHECK(throws([&] { percentile(four, 0); }));
  CHECK(throws([&] { percentile(four, 101); }));
}

static SearchResult result(QueryId q, std::vector<VectorId> ids) {
  SearchResult r;
  r.query_id = q;
  r.ids = ids;
  r.distances.assign(ids.size(), 1.0f);
  return r;
}

static void test_recall() {
  IdMatrix gt;
  gt.n = 2;
  gt.d = 3;
  gt.data = {2, 9, 1, 4, 5, 6};
  const std::vector<SearchResult> found = {result(0, {1, 2}),
                                           result(1, {5, 4})};
  CHECK(recall_at_k(found, gt, 2) == 0.75); // id 1 is not in the top 2
  CHECK(recall_at_k(found, gt, 3) == 4.0 / 6.0);
}

// FNV-1a written out from its definition.
static uint64_t fnv_by_hand(const std::vector<unsigned char> &bytes) {
  uint64_t hash = 0xcbf29ce484222325ull;
  for (unsigned char b : bytes) {
    hash ^= b;
    hash *= 0x100000001b3ull;
  }
  return hash;
}

static void test_result_hash() {
  std::vector<SearchResult> found = {result(0, {7, 300}), result(1, {65537})};
  found[0].distance_evals = 5;
  found[0].hops = 2;
  found[1].distance_evals = 70000;
  found[1].hops = 0;
  // Little-endian uint32 values: 2 queries | query 0, 2 results, 7, 300,
  // 5 evals, 2 hops | query 1, 1 result, 65537, 70000 evals, 0 hops.
  const std::vector<unsigned char> bytes = {
      2,  0, 0, 0, 0, 0, 0, 0, 2,   0,  0, 0, 7, 0, 0, 0,
      44, 1, 0, 0, 5, 0, 0, 0, 2,   0,  0, 0, 1, 0, 0, 0,
      1,  0, 0, 0, 1, 0, 1, 0, 112, 17, 1, 0, 0, 0, 0, 0};
  const std::string expected = result_hash(found);
  CHECK(expected == hex64(fnv_by_hand(bytes)));
  CHECK(expected == "078733fbeb742fe8");
  CHECK(hex64(1) == "0000000000000001");

  // Changes with any id, evals or hops; not with distances or batches.
  using List = std::vector<SearchResult>;
  auto hash_after = [&](void (*change)(List &)) {
    List copy = found;
    change(copy);
    return result_hash(copy);
  };
  CHECK(hash_after([](List &c) { c[1].ids[0] = 65538; }) != expected);
  CHECK(hash_after([](List &c) { c[0].distance_evals = 6; }) != expected);
  CHECK(hash_after([](List &c) { c[1].hops = 1; }) != expected);
  CHECK(hash_after([](List &c) { c[0].distances[1] = 9.5f; }) == expected);
  CHECK(hash_after([](List &c) { c[0].batches = 4; }) == expected);
  CHECK(throws([&] {
    hash_after([](List &c) { c[0].distance_evals = uint64_t(1) << 32; });
  }));
  CHECK(throws([&] { hash_after([](List &c) { std::swap(c[0], c[1]); }); }));
}

static size_t count_fields(const std::string &line) {
  return size_t(std::count(line.begin(), line.end(), ',')) + 1;
}

static bool ends_with(const std::string &text, const std::string &end) {
  return text.size() >= end.size() &&
         text.compare(text.size() - end.size(), end.size(), end) == 0;
}

static void test_columns() {
  // Every mode writes the same 61 columns; cpu rows leave the 25 from
  // backend to from_device_us empty except verified.
  ReportRow row;
  row.mode = "cpu";
  row.start_time_utc = "T";
  row.command = "C";
  CHECK(count_fields(csv_header()) == 61);
  CHECK(count_fields(csv_line(row)) == 61);
  CHECK(csv_header().compare(0, 13, "build_id,tag,") == 0);
  CHECK(csv_header().find(",compiler,build_type,cxx_flags,logical_cpus,") !=
        std::string::npos);
  CHECK(csv_header().find(",result_hash,backend,stage_timing,verified,") !=
        std::string::npos);
  CHECK(ends_with(csv_header(),
                  ",kernel_us,from_device_us,start_time_utc,command\n"));
  CHECK(ends_with(csv_line(row), ",,,no" + std::string(22, ',') + ",T,C\n"));
  // With --stage-timing, submit and wait are empty.
  row.mode = "hybrid-sync";
  row.backend = "cuda";
  row.stage_timing = true;
  row.verified = true;
  CHECK(count_fields(csv_line(row)) == 61);
  CHECK(ends_with(csv_line(row), ",0.00,0.00,0.00,,,0.00,0.00,0.00,T,C\n"));
  CHECK(csv_line(row).find(",cuda,1,yes,,,,auto,0,0,0,0.0000,0,") !=
        std::string::npos);
}

// Two queries as V1 reports them, and the counts that match.
static std::vector<SearchResult> v1_results(VerifyCounts &c) {
  std::vector<SearchResult> v1 = {result(0, {4, 7}), result(1, {9, 2})};
  for (SearchResult &r : v1) {
    r.distance_evals = 30;
    r.hops = 5;
    r.batches = 6;
  }
  c.queries = 2;
  c.k = 2;
  c.ef = 8;
  c.n = 100;
  c.degree = 16;
  c.max_batch = 16;
  c.round_trips = 12;
  c.candidates = 60;
  return v1;
}

static void test_compare_results() {
  VerifyCounts c;
  const std::vector<SearchResult> v1 = v1_results(c);
  CHECK(compare_results(v1, v1, c).empty());

  // The first failing check, after one change to V2's results.
  using List = std::vector<SearchResult>;
  auto failure_after = [&](void (*change)(List &)) {
    List v2 = v1;
    change(v2);
    return compare_results(v2, v1, c);
  };
  CHECK(failure_after([](List &v) { v[1].ids[0] = 3; }) ==
        "query 1: ids differ");
  CHECK(failure_after([](List &v) {
          v[0].distances[1] = std::nextafter(v[0].distances[1], 2.0f);
        }) == "query 0: distance 1 differs");
  CHECK(failure_after([](List &v) { v[0].distance_evals = 31; }) ==
        "query 0: distance_evals differ");
  CHECK(failure_after([](List &v) { v[1].hops = 4; }) ==
        "query 1: hops differ");
  CHECK(failure_after([](List &v) { v[1].batches = 7; }) ==
        "query 1: batches differ");
  CHECK(failure_after([](List &v) { v.pop_back(); }) ==
        "compared 1 V2 and 2 V1 queries, expected 2");

  // Failures with V2 equal to V1.
  List empty = v1;
  empty[0].ids.clear();
  empty[0].distances.clear();
  CHECK(compare_results(empty, empty, c) ==
        "query 0: a result list does not have 2 entries");
  CHECK(compare_results({}, {}, VerifyCounts()) ==
        "compared 0 V2 and 0 V1 queries, expected 0");
  VerifyCounts wrong = c;
  wrong.candidates = 59;
  CHECK(compare_results(v1, v1, wrong) ==
        "the backends received 59 candidates but the results count 60 "
        "evaluations");
  wrong = c;
  wrong.round_trips = 13;
  CHECK(compare_results(v1, v1, wrong) ==
        "the backends made 13 round trips but the results count 12 batches");
  wrong = c;
  wrong.max_batch = 15;
  CHECK(compare_results(v1, v1, wrong) ==
        "max_batch 15 is below the degree 16, so a batch could take two "
        "round trips");
}

// verify_against_v1 runs V1 itself and compares the V2 results it is given
// with it, so changed results must fail.
static void test_verify_against_v1(const std::string &dir) {
  Options o;
  o.data = dir;
  const Dataset data = load_dataset(o);
  SearchConfig config;
  config.ef_search = 32;
  config.cpu_threads = 2;
  const std::vector<SearchResult> v1 =
      search_all_cpu(data.graph, data.base, data.queries, config);
  BackendTotals totals;
  for (const SearchResult &r : v1) {
    totals.round_trips += r.batches;
    totals.candidates += r.distance_evals;
  }
  const std::vector<std::string> hashes(2, result_hash(v1));
  const uint32_t degree = data.graph.degree;
  CHECK(verify_against_v1(data, config, degree, hashes, v1, totals).empty());

  std::vector<SearchResult> changed = v1;
  changed[5].hops += 1;
  CHECK(verify_against_v1(data, config, degree, hashes, changed, totals) ==
        "query 5: hops differ");
  CHECK(
      verify_against_v1(data, config, degree, {hashes[0], hex64(1)}, v1, totals)
          .find("repetition 2: ") == 0);
  CHECK(verify_against_v1(data, config, degree, {}, v1, totals) ==
        "no repetition to compare");
}

static std::string read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

static void write_file(const std::string &path, const std::string &text) {
  std::ofstream(path, std::ios::binary) << text;
}

static void test_csv_files(const std::string &dir) {
  const std::string header = "a,b\n";
  const std::string path = dir + "/out.csv";
  std::filesystem::remove(path);
  append_csv(path, header, {"1,2\n"}); // missing file: header first
  CHECK(read_file(path) == "a,b\n1,2\n");
  append_csv(path, header, {"3,4\n", "5,6\n"}); // same header: rows only
  CHECK(read_file(path) == "a,b\n1,2\n3,4\n5,6\n");

  write_file(path, ""); // empty file: header first
  check_csv(path, header);
  append_csv(path, header, {"1,2\n"});
  CHECK(read_file(path) == "a,b\n1,2\n");

  // Refused, with nothing written.
  for (const char *text :
       {"a,c\n1,2\n", "a,b,c\n1,2,3\n", "a\n", "\n", "a,b", "a,b\n1,"}) {
    write_file(path, text);
    CHECK(throws([&] { check_csv(path, header); }));
    CHECK(throws([&] { append_csv(path, header, {"1,2\n"}); }));
    CHECK(read_file(path) == text);
  }

  // A missing file is not left behind; a path that cannot be opened fails.
  std::filesystem::remove(path);
  check_csv(path, header);
  CHECK(!std::filesystem::exists(path));
  CHECK(throws([&] { check_csv(dir + "/missing/out.csv", header); }));
  CHECK(!std::filesystem::exists(dir + "/missing"));
}

// The fields of one CSV line.
static std::vector<std::string> split(const std::string &line) {
  std::vector<std::string> fields(1);
  for (char c : line) {
    if (c == ',')
      fields.emplace_back();
    else if (c != '\n')
      fields.back() += c;
  }
  return fields;
}

// The value of a column in a line written by csv_line.
static std::string field(const std::string &line, const std::string &name) {
  const std::vector<std::string> names = split(csv_header());
  const std::vector<std::string> values = split(line);
  const size_t i =
      size_t(std::find(names.begin(), names.end(), name) - names.begin());
  CHECK(i < names.size() && values.size() == names.size());
  return values[i];
}

// A constructed pass: 1000 queries with latencies of 1 to 1000 us in shuffled
// order, on 4 threads in 0.25 s.
static void test_describe_pass() {
  Pass pass;
  for (uint32_t q = 0; q < 1000; ++q) {
    pass.results.push_back(result(q, {}));
    pass.latencies_us.push_back(double((q * 7919) % 1000 + 1));
  }
  pass.threads_used = 4;
  pass.seconds = 0.25;
  Dataset data;
  data.gt.n = 1000;
  data.gt.d = 1;
  data.gt.data.assign(1000, 0);
  ReportRow row;
  row.mode = "cpu";
  row.k = 1;
  describe_pass(row, pass, data);
  const std::string line = csv_line(row);
  CHECK(field(line, "qps") == "4000.0");
  CHECK(field(line, "latency_mean_us") == "500.50");
  CHECK(field(line, "latency_p50_us") == "500.00");
  CHECK(field(line, "latency_p95_us") == "950.00");
  CHECK(field(line, "latency_p99_us") == "990.00");
  CHECK(field(line, "threads_used") == "4");
}

// How the test backends below misbehave.
struct Fault {
  QueryId query = UINT32_MAX; // a wrong distance for this query and id,
  VectorId id = INVALID_ID;
  uint32_t pass = 0;        // in this pass of the launch (from 1), or all if 0
  bool fail = false;        // throw on every call
  bool drop_counts = false; // report no calls or candidates
};

// Computes distances on the CPU, except for the fault. Counts the candidates
// of every pass.
class TestBackend : public DistanceBackend {
public:
  TestBackend(const FloatMatrix &base, const Fault &fault, const uint32_t &pass)
      : cpu(base), fault_(fault), pass_(pass) {}

  void set_queries(const float *queries, uint32_t num_queries,
                   uint32_t dim) override {
    cpu.set_queries(queries, num_queries, dim);
  }

  void compute_distances(const CandidateBatch &candidates,
                         std::vector<float> &distances) override {
    if (fault_.fail)
      throw std::runtime_error("test backend failed");
    cpu.compute_distances(candidates, distances);
    all_candidates += candidates.size();
    if (fault_.pass != 0 && fault_.pass != pass_)
      return;
    for (size_t i = 0; i < candidates.size(); ++i)
      if (candidates.query_ids[i] == fault_.query &&
          candidates.vector_ids[i] == fault_.id)
        distances[i] = 1e30f;
  }

  CpuDistanceBackend cpu;
  uint64_t all_candidates = 0; // never reset

private:
  const Fault &fault_;
  const uint32_t &pass_;
};

// One test backend per thread.
struct TestBackends : LaunchBackends {
  TestBackends(const Dataset &data, uint32_t threads, const Fault &f)
      : fault(f) {
    for (uint32_t t = 0; t < threads; ++t) {
      all.push_back(std::make_unique<TestBackend>(data.base, fault, passes));
      all.back()->set_queries(data.queries.data.data(), data.queries.n,
                              data.queries.d);
      slots.push_back(all.back().get());
    }
  }
  void reset_stats() override {
    ++passes;
    for (const auto &b : all)
      b->cpu.reset_counts();
  }
  BackendTotals read_stats() const override {
    BackendTotals t;
    for (const auto &b : all)
      if (!fault.drop_counts) {
        t.round_trips += b->cpu.calls();
        t.candidates += b->cpu.candidates();
      }
    return t;
  }

  Fault fault;
  uint32_t passes = 0; // one reset per pass
  std::vector<std::unique_ptr<TestBackend>> all;
};

// The ids whose distances a search for query q at ef computes.
static std::vector<VectorId> evaluated(const Dataset &data, QueryId q,
                                       uint32_t ef) {
  QuerySearch search(data.graph, ef, 10);
  std::vector<VectorId> ids, all;
  std::vector<float> distances;
  search.begin();
  while (search.next(ids)) {
    distances.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
      distances[i] =
          l2(data.queries.row(q), data.base.row(ids[i]), data.base.d);
    all.insert(all.end(), ids.begin(), ids.end());
    search.absorb(ids, distances.data());
  }
  return all;
}

// Whole hybrid-sync launches through run_benchmark, with test backends.
static void test_launches(const std::string &scratch, const std::string &dir) {
  Options o;
  o.data = dir;
  o.mode = "hybrid-sync";
  o.backend = "cpu";
  o.verify = true;
  o.ef = {16, 64};
  o.threads = 2;
  o.repetitions = 2;
  o.csv = scratch + "/launch.csv";
  const Dataset data = load_dataset(o);
  // Runs one launch; returns its error, or "" if it succeeded.
  const auto launch = [&](TestBackends &b) -> std::string {
    std::filesystem::remove(o.csv);
    try {
      run_benchmark(o, data, launch_row(o, data), b);
    } catch (const std::exception &e) {
      CHECK(!std::filesystem::exists(o.csv));
      return e.what();
    }
    return "";
  };
  // The column of every row of the CSV.
  const auto column = [&](const std::string &name) {
    std::vector<std::string> values;
    std::ifstream in(o.csv);
    std::string line;
    std::getline(in, line);
    CHECK(line + "\n" == csv_header());
    while (std::getline(in, line))
      values.push_back(field(line, name));
    return values;
  };
  const std::vector<std::string> yes(4, "yes"), no(4, "no");

  // Correct backends: every pass runs, warm-ups included, and all of it.
  TestBackends good(data, 2, Fault());
  CHECK(launch(good).empty() && column("verified") == yes);
  CHECK(good.passes == 2 * (1 + 2));
  uint64_t candidates = 0, expected = 0;
  for (const auto &b : good.all)
    candidates += b->all_candidates;
  SearchConfig config;
  config.cpu_threads = 2;
  for (uint32_t ef : o.ef) {
    config.ef_search = ef;
    for (const SearchResult &r :
         search_all_cpu(data.graph, data.base, data.queries, config))
      expected += 3 * r.distance_evals;
  }
  CHECK(candidates == expected);
  o.verify = false;
  TestBackends unverified(data, 2, Fault());
  CHECK(launch(unverified).empty() && column("verified") == no);
  o.verify = true;

  // A wrong distance that changes a result at ef 64 but is never computed at
  // ef 16: a result id at ef 64 that the search at ef 16 does not evaluate.
  config.ef_search = 64;
  const std::vector<SearchResult> v1 =
      search_all_cpu(data.graph, data.base, data.queries, config);
  Fault wrong;
  for (QueryId q = 0; q < v1.size() && wrong.id == INVALID_ID; ++q) {
    const std::vector<VectorId> early = evaluated(data, q, 16);
    for (VectorId id : v1[q].ids)
      if (std::count(early.begin(), early.end(), id) == 0) {
        wrong.query = q;
        wrong.id = id;
      }
  }
  CHECK(wrong.id != INVALID_ID);
  TestBackends late(data, 2, wrong);
  CHECK(launch(late).find("ef=64: --verify failed: ") == 0);
  // The same fault in the first timed pass at ef 64 only (passes 4 to 6).
  wrong.pass = 5;
  TestBackends once(data, 2, wrong);
  CHECK(launch(once).find("ef=64: repetitions gave different results") == 0);

  Fault fail;
  fail.fail = true;
  TestBackends failing(data, 2, fail);
  CHECK(launch(failing) == "test backend failed");
  Fault drop;
  drop.drop_counts = true;
  TestBackends uncounted(data, 2, drop);
  CHECK(launch(uncounted).find("ef=16: --verify failed: the backends "
                               "received 0 candidates") == 0);
}

// The visit counts of two disjoint halves of the learn rows add up to those
// of all of them.
static void test_visit_counts(const std::string &scratch,
                              const std::string &dir) {
  Options o;
  o.data = dir;
  const Dataset data = load_dataset(o);
  // The generated sets have no learn.fbin; their queries serve as one.
  std::filesystem::copy_file(dir + "/query.fbin", scratch + "/learn.fbin",
                             std::filesystem::copy_options::overwrite_existing);
  o.data = scratch;
  o.threads = 2;
  o.ef = {32};
  o.visit_counts = scratch + "/visits.bin";
  const auto counts = [&](uint32_t first, uint32_t count) {
    o.visit_first = first;
    o.visit_count = count;
    write_visit_counts(o, data);
    const std::string bytes = read_file(o.visit_counts);
    std::vector<uint32_t> values(bytes.size() / 4);
    std::memcpy(values.data(), bytes.data(), values.size() * 4);
    return values;
  };
  const std::vector<uint32_t> all = counts(0, 0), low = counts(0, 50),
                              high = counts(50, 78);
  CHECK(all.size() == 2 + data.base.n && all[0] == data.base.n && all[1] == 1);
  CHECK(low != all && high != all);
  for (size_t i = 2; i < all.size(); ++i)
    CHECK(low[i] + high[i] == all[i]);
  CHECK(throws([&] { counts(50, 79); }));
  CHECK(throws([&] { counts(128, 0); }));
}

int main(int argc, char **argv) {
  CHECK(argc == 3); // scratch directory, data directory
  std::filesystem::create_directories(argv[1]);
  test_options();
  test_percentile();
  test_recall();
  test_result_hash();
  test_columns();
  test_describe_pass();
  test_compare_results();
  test_verify_against_v1(argv[2]);
  test_csv_files(argv[1]);
  test_launches(argv[1], argv[2]);
  test_visit_counts(argv[1], argv[2]);
  std::printf("test_cli passed\n");
  return 0;
}
