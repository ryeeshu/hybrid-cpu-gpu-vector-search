/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "hvs/cpu_backend.h"
#include "hvs/io.h"
#include "hvs/search.h"

#ifdef HVS_CUDA
#include "hvs/gpu/cuda_backend.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <omp.h>

using namespace hvs;

constexpr uint32_t MANY_THREADS = 8;
// Time a call with Fault::SLOW spends waiting.
constexpr std::chrono::milliseconds SLOW_CALL(5);

struct Dataset {
  FloatMatrix base, queries;
  Graph graph;
};

// Sums over all queries of a run.
struct Totals {
  uint64_t evals = 0, hops = 0, batches = 0;
  explicit Totals(const std::vector<SearchResult> &results) {
    for (const SearchResult &r : results) {
      evals += r.distance_evals;
      hops += r.hops;
      batches += r.batches;
    }
  }
};

// Number of queries whose results differ in any field, distances bit for bit.
// Result lists of different lengths count as all different.
static size_t count_mismatches(const std::vector<SearchResult> &a,
                               const std::vector<SearchResult> &b) {
  if (a.size() != b.size())
    return std::max(a.size(), b.size());
  size_t mismatches = 0;
  for (size_t q = 0; q < a.size(); ++q) {
    const SearchResult &x = a[q], &y = b[q];
    const bool same_distances =
        x.distances.size() == y.distances.size() &&
        (x.distances.empty() ||
         std::memcmp(x.distances.data(), y.distances.data(),
                     x.distances.size() * sizeof(float)) == 0);
    if (x.query_id != y.query_id || x.ids != y.ids || !same_distances ||
        x.distance_evals != y.distance_evals || x.hops != y.hops ||
        x.batches != y.batches)
      ++mismatches;
  }
  return mismatches;
}

// Number of queries whose ids, distance evaluations or hops differ from the
// reference files.
static size_t count_reference_mismatches(const std::vector<SearchResult> &found,
                                         const IdMatrix &ref_ids,
                                         const IdMatrix &ref_work) {
  if (found.size() != ref_ids.n)
    return std::max<size_t>(found.size(), ref_ids.n);
  size_t mismatches = 0;
  for (uint32_t q = 0; q < ref_ids.n; ++q) {
    const SearchResult &r = found[q];
    const std::vector<VectorId> ids(ref_ids.row(q), ref_ids.row(q) + ref_ids.d);
    if (r.query_id != q || r.ids != ids ||
        r.distance_evals != ref_work.row(q)[0] || r.hops != ref_work.row(q)[1])
      ++mismatches;
  }
  return mismatches;
}

// Shared by the backends of one run. Every call waits until min_threads
// distinct threads have called some backend. A thread waiting in its first
// call takes no more queries, so with min_threads equal to the thread count
// every thread gets queries, and all are inside backends at the same time.
struct Rendezvous {
  uint32_t min_threads = 1;
  std::mutex mutex;
  std::condition_variable arrived;
  std::set<std::thread::id> callers;
};

enum class Fault { NONE, PERTURB, THROW, SLOW };

// Spins for at least the given time.
static void busy_wait(std::chrono::microseconds time) {
  const auto start = std::chrono::steady_clock::now();
  while (std::chrono::steady_clock::now() - start < time)
    ;
}

// Wraps another backend and records a violation if it is called from a
// second thread or entered concurrently. A fault spoils the pair (query,
// vector): PERTURB raises its distance by one ulp, THROW makes the call throw
// and SLOW makes the call take SLOW_CALL longer.
class TestBackend : public DistanceBackend {
public:
  TestBackend(DistanceBackend &inner, Rendezvous &rendezvous, Fault fault,
              QueryId query, VectorId vector)
      : inner_(inner), rendezvous_(rendezvous), fault_(fault), query_(query),
        vector_(vector) {}

  void set_queries(const float *queries, uint32_t num_queries,
                   uint32_t dim) override {
    inner_.set_queries(queries, num_queries, dim);
  }

  void compute_distances(const CandidateBatch &candidates,
                         std::vector<float> &distances) override {
    if (inside_.fetch_add(1) != 0)
      violated_ = true;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (owner_ == std::thread::id())
        owner_ = std::this_thread::get_id();
      else if (owner_ != std::this_thread::get_id())
        violated_ = true;
    }
    wait_for_threads();
    inner_.compute_distances(candidates, distances);
    for (size_t i = 0; i < candidates.size(); ++i)
      if (fault_ != Fault::NONE && candidates.query_ids[i] == query_ &&
          candidates.vector_ids[i] == vector_) {
        if (fault_ == Fault::THROW)
          throw std::runtime_error("test failure");
        if (fault_ == Fault::SLOW)
          busy_wait(SLOW_CALL);
        else
          distances[i] = std::nextafter(distances[i], INFINITY);
      }
    inside_.fetch_sub(1);
  }

  bool violated() const { return violated_; }

private:
  void wait_for_threads() {
    Rendezvous &r = rendezvous_;
    std::unique_lock<std::mutex> lock(r.mutex);
    r.callers.insert(std::this_thread::get_id());
    r.arrived.notify_all();
    if (!r.arrived.wait_for(lock, std::chrono::seconds(10),
                            [&] { return r.callers.size() >= r.min_threads; }))
      throw std::runtime_error("too few threads called a backend");
  }

  DistanceBackend &inner_;
  Rendezvous &rendezvous_;
  Fault fault_;
  QueryId query_;
  VectorId vector_;
  std::atomic<int> inside_{0};
  std::atomic<bool> violated_{false};
  std::mutex mutex_;
  std::thread::id owner_; // no thread until the first call
};

// One TestBackend per thread, and the pointers search_all_hybrid_sync takes.
// The rendezvous waits for every thread.
struct TestBackends {
  Rendezvous rendezvous;
  std::vector<std::unique_ptr<CpuDistanceBackend>> inner;
  std::vector<std::unique_ptr<TestBackend>> owned;
  std::vector<DistanceBackend *> slots;

  TestBackends() = default;

  // Wraps new CPU backends that hold the queries.
  TestBackends(const Dataset &data, uint32_t threads, Fault fault = Fault::NONE,
               QueryId query = 0, VectorId vector = 0) {
    for (uint32_t t = 0; t < threads; ++t) {
      inner.push_back(std::make_unique<CpuDistanceBackend>(data.base));
      inner.back()->set_queries(data.queries.data.data(), data.queries.n,
                                data.queries.d);
      add(*inner.back(), fault, query, vector);
    }
  }

  // Wraps backend for the next thread.
  void add(DistanceBackend &backend, Fault fault = Fault::NONE,
           QueryId query = 0, VectorId vector = 0) {
    owned.push_back(std::make_unique<TestBackend>(backend, rendezvous, fault,
                                                  query, vector));
    slots.push_back(owned.back().get());
    rendezvous.min_threads = uint32_t(slots.size());
  }
};

// V2 equals V1 and the reference at every ef and thread count, and the
// backends received exactly the batches and candidates the results report.
static void test_matches(const Dataset &data, const std::string &dir,
                         bool exhausts) {
#ifdef HVS_CUDA
  // Created and given the queries before any search; max_batch = degree
  // keeps every search batch to one round trip.
  CudaBackendOptions options;
  options.max_batch = data.graph.degree;
  options.pinned = true;
  options.resident = false;
  std::vector<std::unique_ptr<CudaDistanceBackend>> gpus;
  for (uint32_t t = 0; t < MANY_THREADS; ++t) {
    gpus.push_back(std::make_unique<CudaDistanceBackend>(data.base, options));
    gpus.back()->set_queries(data.queries.data.data(), data.queries.n,
                             data.queries.d);
  }
#endif
  for (uint32_t ef : {10u, 32u, 64u}) {
    const std::string stem = dir + "/ref_ef" + std::to_string(ef);
    const IdMatrix ref_ids = load_matrix<VectorId>(stem + ".ibin");
    const IdMatrix ref_work = load_matrix<VectorId>(stem + "_work.ibin");
    CHECK(ref_ids.n == data.queries.n && ref_work.n == data.queries.n);
    SearchConfig config;
    config.k = ref_ids.d;
    config.ef_search = ef;
    const auto v1 = search_all_cpu(data.graph, data.base, data.queries, config);
    CHECK(v1.size() == data.queries.n);
    CHECK(count_reference_mismatches(v1, ref_ids, ref_work) == 0);
    const Totals work(v1);
    // Some expansions found nothing new, so they asked for no distances.
    if (exhausts && ef == 64)
      CHECK(work.batches < work.hops + v1.size());

    for (uint32_t threads : {1u, MANY_THREADS}) {
      config.cpu_threads = threads;
      uint32_t used = 0;
      CHECK(count_mismatches(search_all_cpu(data.graph, data.base, data.queries,
                                            config, nullptr, &used),
                             v1) == 0);
      CHECK(used == threads);

      // Every backend is called, only from its own thread.
      TestBackends cpus(data, threads);
      used = 0;
      const auto v2 = search_all_hybrid_sync(data.graph, data.queries, config,
                                             cpus.slots, nullptr, &used);
      CHECK(used == threads);
      CHECK(count_mismatches(v2, v1) == 0);
      CHECK(count_reference_mismatches(v2, ref_ids, ref_work) == 0);
      uint64_t calls = 0, candidates = 0;
      for (uint32_t t = 0; t < threads; ++t) {
        CHECK(!cpus.owned[t]->violated() && cpus.inner[t]->calls() > 0);
        calls += cpus.inner[t]->calls();
        candidates += cpus.inner[t]->candidates();
      }
      CHECK(calls == work.batches && candidates == work.evals);

#ifdef HVS_CUDA
      TestBackends wrapped;
      for (uint32_t t = 0; t < threads; ++t) {
        gpus[t]->reset_stats();
        wrapped.add(*gpus[t]);
      }
      used = 0;
      CHECK(count_mismatches(search_all_hybrid_sync(data.graph, data.queries,
                                                    config, wrapped.slots,
                                                    nullptr, &used),
                             v1) == 0);
      CHECK(used == threads);
      uint64_t round_trips = 0;
      candidates = 0;
      for (uint32_t t = 0; t < threads; ++t) {
        const CudaBackendStats &s = gpus[t]->stats();
        CHECK(!wrapped.owned[t]->violated() && s.round_trips > 0);
        round_trips += s.round_trips;
        candidates += s.candidates;
      }
      CHECK(round_trips == work.batches && candidates == work.evals);
#endif
    }
  }
}

// Controls for the checks above, and errors raised inside the parallel
// region, which must reach the caller as exceptions.
static void test_controls_and_errors(const Dataset &data) {
  SearchConfig config;
  config.ef_search = 10;
  config.cpu_threads = MANY_THREADS;

  // One instance in every slot must be caught. The rendezvous holds the
  // first caller inside it until a second thread calls, on every run.
  TestBackends one(data, 1);
  one.rendezvous.min_threads = 2;
  search_all_hybrid_sync(
      data.graph, data.queries, config,
      std::vector<DistanceBackend *>(MANY_THREADS, one.slots[0]));
  CHECK(one.owned[0]->violated());

  // One distance one ulp off, in the result of one query, must show up as
  // exactly one mismatching query.
  const auto v1 = search_all_cpu(data.graph, data.base, data.queries, config);
  const QueryId q = data.queries.n / 2;
  TestBackends perturbed(data, MANY_THREADS, Fault::PERTURB, q, v1[q].ids[0]);
  CHECK(count_mismatches(search_all_hybrid_sync(data.graph, data.queries,
                                                config, perturbed.slots),
                         v1) == 1);

  TestBackends failing(data, MANY_THREADS, Fault::THROW, data.queries.n / 3,
                       data.graph.entry_point);
  bool threw = false;
  try {
    search_all_hybrid_sync(data.graph, data.queries, config, failing.slots);
  } catch (const std::runtime_error &e) {
    threw = std::string(e.what()) == "test failure";
  }
  CHECK(threw);

  threw = false;
  try {
    QuerySearch search(data.graph, 0, 10);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);

  // Thrown while each thread sets up, before any query runs.
  config.ef_search = 0;
  threw = false;
  try {
    search_all_cpu(data.graph, data.base, data.queries, config);
  } catch (const std::invalid_argument &) {
    threw = true;
  }
  CHECK(threw);
}

// Each query's latency lands in its own slot, in microseconds. Only lower
// bounds are exact for one query; the total is bounded by the wall time of
// the call on every thread.
static void test_latencies(const Dataset &data) {
  SearchConfig config;
  config.ef_search = 10;
  config.cpu_threads = MANY_THREADS;
  const QueryId slow = data.queries.n / 2;
  TestBackends backends(data, MANY_THREADS, Fault::SLOW, slow,
                        data.graph.entry_point);
  for (bool hybrid : {false, true}) {
    std::vector<double> latencies;
    uint32_t used = 0;
    const auto start = std::chrono::steady_clock::now();
    if (hybrid)
      search_all_hybrid_sync(data.graph, data.queries, config, backends.slots,
                             &latencies, &used);
    else
      search_all_cpu(data.graph, data.base, data.queries, config, &latencies,
                     &used);
    const std::chrono::duration<double, std::micro> wall =
        std::chrono::steady_clock::now() - start;
    CHECK(latencies.size() == data.queries.n);
    double sum = 0;
    for (double us : latencies) {
      CHECK(us > 0);
      sum += us;
    }
    CHECK(sum <= wall.count() * used);
    if (hybrid)
      CHECK(latencies[slow] >= 1000.0 * SLOW_CALL.count());
  }
}

// A search started inside another parallel region gets one thread when only
// one level may be active, and must report that.
static void test_nested(const Dataset &data) {
  SearchConfig config;
  config.cpu_threads = MANY_THREADS;
  const int levels = omp_get_max_active_levels();
  omp_set_max_active_levels(1);
  uint32_t used[2] = {0, 0};
#pragma omp parallel num_threads(2)
  search_all_cpu(data.graph, data.base, data.queries, config, nullptr,
                 &used[omp_get_thread_num()]);
  omp_set_max_active_levels(levels);
  CHECK(used[0] == 1 && used[1] == 1);
}

int main(int argc, char **argv) {
  // Data directory, then --exhausts if its searches must include expansions
  // that find nothing new.
  CHECK(argc == 2 || (argc == 3 && std::string(argv[2]) == "--exhausts"));
  const std::string dir = argv[1];
  const Dataset data{load_matrix<float>(dir + "/base.fbin"),
                     load_matrix<float>(dir + "/query.fbin"),
                     load_graph(dir + "/graph0.bin", dir + "/upper.bin")};
  // Queries go out in chunks of 16, and each thread of a run must get one.
  CHECK(data.queries.n >= 16 * MANY_THREADS);

  test_matches(data, dir, argc == 3);
  test_controls_and_errors(data);
  test_latencies(data);
  test_nested(data);

  std::printf("test_hybrid passed\n");
  return 0;
}
