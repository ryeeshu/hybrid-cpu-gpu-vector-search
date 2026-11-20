/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "hvs/cpu_backend.h"
#include "hvs/gpu/cuda_backend.h"
#include "hvs/io.h"
#include "hvs/search.h"

#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace hvs;

// Device bytes a backend should hold. Every buffer has 4-byte elements: ids
// and results for max_batch pairs, the staging buffer or the resident base,
// and the queries.
static uint64_t expected_device_bytes(const FloatMatrix &base,
                                      uint32_t num_queries,
                                      const CudaBackendOptions &options) {
  const uint64_t batch = options.max_batch;
  const uint64_t vectors =
      options.resident ? batch + uint64_t(base.n) * base.d : batch * base.d;
  return 4 * (2 * batch + vectors + uint64_t(num_queries) * base.d);
}

// The synthetic sets are integer-valued, so GPU and CPU distances must be
// equal bit for bit.
static void test_matches_cpu(const FloatMatrix &base,
                             const FloatMatrix &queries,
                             const CudaBackendOptions &options) {
  CpuDistanceBackend cpu(base);
  CudaDistanceBackend gpu(base, options);
  cpu.set_queries(queries.data.data(), queries.n, queries.d);
  gpu.set_queries(queries.data.data(), queries.n, queries.d);

  std::mt19937 rng(7);
  std::vector<float> expected, found;
  uint64_t round_trips = 0;
  // Sizes around warp, block and max_batch boundaries; 0 is an empty batch.
  for (uint32_t count :
       {0u, 1u, 31u, 32u, 33u, 255u, 256u, 257u, options.max_batch - 1,
        options.max_batch, options.max_batch + 1, 3 * options.max_batch + 5}) {
    round_trips += (count + options.max_batch - 1) / options.max_batch;
    CandidateBatch batch;
    for (uint32_t i = 0; i < count; ++i)
      batch.push(rng() % queries.n, rng() % base.n);
    cpu.compute_distances(batch, expected);
    found.assign(3, -1.0f); // stale contents must be replaced
    gpu.compute_distances(batch, found);
    CHECK(found.size() == count);
    for (uint32_t i = 0; i < count; ++i)
      CHECK(found[i] == expected[i]);
  }

  // Accounting: every candidate ships its ids, plus its vector when staged.
  const CudaBackendStats &s = gpu.stats();
  const uint64_t per_candidate =
      options.resident ? 8 : 4 + uint64_t(base.d) * sizeof(float);
  CHECK(s.bytes_to_device == s.candidates * per_candidate);
  CHECK(s.bytes_from_device == s.candidates * sizeof(float));
  CHECK(s.round_trips == round_trips);
  CHECK(gpu.device_bytes() == expected_device_bytes(base, queries.n, options));

  CandidateBatch bad;
  bad.push(0, base.n); // vector id out of range
  bool threw = false;
  try {
    gpu.compute_distances(bad, found);
  } catch (const std::out_of_range &) {
    threw = true;
  }
  CHECK(threw);
}

// Checks one batch of random pairs against the CPU backend.
static void check_against_cpu(CudaDistanceBackend &gpu, const FloatMatrix &base,
                              const FloatMatrix &queries) {
  CpuDistanceBackend cpu(base);
  cpu.set_queries(queries.data.data(), queries.n, queries.d);
  std::mt19937 rng(11);
  CandidateBatch batch;
  for (uint32_t i = 0; i < 1000; ++i)
    batch.push(rng() % queries.n, rng() % base.n);
  std::vector<float> expected, found;
  cpu.compute_distances(batch, expected);
  gpu.compute_distances(batch, found);
  CHECK(found == expected);
}

// Random integer-valued matrix, so GPU and CPU distances are equal exactly.
static FloatMatrix random_matrix(uint32_t n, uint32_t d, std::mt19937 &rng) {
  FloatMatrix m;
  m.n = n;
  m.d = d;
  m.data.resize(size_t(n) * d);
  for (float &x : m.data)
    x = float(rng() % 16);
  return m;
}

// Regression test for a failed set_queries: the backend must hold no queries
// afterwards, whether or not it held some before, and the error must not be
// reported again by a later call on this or another backend.
static void test_failed_set_queries() {
  // Wide vectors, so UINT32_MAX queries need about 17.6 TB of device memory;
  // under WSL, requests of a few hundred GB take seconds to fail.
  std::mt19937 rng(5);
  const FloatMatrix base = random_matrix(500, 1024, rng);
  const FloatMatrix queries = random_matrix(100, 1024, rng);
  CudaBackendOptions options;
  CudaDistanceBackend gpu(base, options);
  // Makes set_queries fail, then checks that no queries are registered.
  const auto fail_set_queries = [&] {
    bool threw = false;
    try {
      // The host array is not read before the allocation fails.
      gpu.set_queries(queries.data.data(), UINT32_MAX, queries.d);
    } catch (const std::runtime_error &) {
      threw = true;
    }
    CHECK(threw);
    CHECK(gpu.device_bytes() == expected_device_bytes(base, 0, options));

    CandidateBatch batch;
    batch.push(0, 0);
    std::vector<float> found;
    threw = false;
    try {
      gpu.compute_distances(batch, found);
    } catch (const std::out_of_range &) {
      threw = true;
    }
    CHECK(threw);
  };

  fail_set_queries();
  gpu.set_queries(queries.data.data(), queries.n, queries.d);
  check_against_cpu(gpu, base, queries);
  CHECK(gpu.device_bytes() == expected_device_bytes(base, queries.n, options));

  // A second set_queries replaces the first.
  const FloatMatrix other = random_matrix(50, base.d, rng);
  gpu.set_queries(other.data.data(), other.n, other.d);
  check_against_cpu(gpu, base, other);
  CHECK(gpu.device_bytes() == expected_device_bytes(base, other.n, options));

  // A failure after a success drops the old queries too.
  fail_set_queries();
  gpu.set_queries(queries.data.data(), queries.n, queries.d);
  check_against_cpu(gpu, base, queries);
  CHECK(gpu.device_bytes() == expected_device_bytes(base, queries.n, options));

  CudaDistanceBackend fresh(base, options);
  fresh.set_queries(queries.data.data(), queries.n, queries.d);
  check_against_cpu(fresh, base, queries);
}

// Runs rounds batches of exactly max_batch pairs after one warm-up batch and
// returns the totals. Times are checked only as totals over many round trips.
static CudaBackendStats run_batches(const FloatMatrix &base,
                                    const FloatMatrix &queries,
                                    const CudaBackendOptions &options,
                                    uint32_t rounds) {
  CudaDistanceBackend gpu(base, options);
  gpu.set_queries(queries.data.data(), queries.n, queries.d);
  std::mt19937 rng(3);
  CandidateBatch batch;
  for (uint32_t i = 0; i < options.max_batch; ++i)
    batch.push(rng() % queries.n, rng() % base.n); // ids may repeat
  std::vector<float> found;
  gpu.compute_distances(batch, found);
  gpu.reset_stats();
  for (uint32_t r = 0; r < rounds; ++r)
    gpu.compute_distances(batch, found);
  return gpu.stats();
}

// The recorded intervals are disjoint parts of each round trip and together
// cover most of it.
static void check_times(const CudaBackendStats &s, bool stage_timing,
                        uint32_t rounds) {
  CHECK(s.round_trips == rounds);
  CHECK(s.gather_us > 0);
  double sum = s.gather_us;
  if (stage_timing) {
    CHECK(s.to_device_us > 0 && s.kernel_us > 0 && s.from_device_us > 0);
    CHECK(s.submit_us == 0 && s.wait_us == 0);
    sum += s.to_device_us + s.kernel_us + s.from_device_us;
  } else {
    CHECK(s.to_device_us == 0 && s.kernel_us == 0 && s.from_device_us == 0);
    CHECK(s.submit_us > 0 && s.wait_us > 0);
    sum += s.submit_us + s.wait_us;
  }
  CHECK(sum <= s.round_trip_us);
  CHECK(s.round_trip_us - sum <= 0.25 * s.round_trip_us);
}

// The synchronous hybrid loop on GPU distances must reproduce the reference
// search exactly: same ids, distance evaluations and hops.
static void test_search_matches_reference(const std::string &dir,
                                          const FloatMatrix &base,
                                          const FloatMatrix &queries,
                                          const CudaBackendOptions &options) {
  const Graph graph = load_graph(dir + "/graph0.bin", dir + "/upper.bin");
  CudaDistanceBackend gpu(base, options);
  gpu.set_queries(queries.data.data(), queries.n, queries.d);
  for (uint32_t ef : {10u, 64u}) {
    const std::string stem = dir + "/ref_ef" + std::to_string(ef);
    const IdMatrix ref_ids = load_matrix<VectorId>(stem + ".ibin");
    const IdMatrix ref_work = load_matrix<VectorId>(stem + "_work.ibin");
    QuerySearch search(graph, ef, ref_ids.d);
    CandidateBatch batch;
    std::vector<float> distances;
    SearchResult r;
    for (uint32_t q = 0; q < queries.n; ++q) {
      search_with_backend(search, gpu, q, batch, distances, r);
      CHECK(r.ids.size() == ref_ids.d);
      for (uint32_t i = 0; i < ref_ids.d; ++i)
        CHECK(r.ids[i] == ref_ids.row(q)[i]);
      CHECK(r.distance_evals == ref_work.row(q)[0]);
      CHECK(r.hops == ref_work.row(q)[1]);
    }
  }
}

int main(int argc, char **argv) {
  CHECK(argc == 2); // synthetic data directory
  const std::string dir = argv[1];
  const FloatMatrix base = load_matrix<float>(dir + "/base.fbin");
  const FloatMatrix queries = load_matrix<float>(dir + "/query.fbin");

  for (bool resident : {false, true})
    for (bool pinned : {false, true})
      for (bool stage_timing : {false, true}) {
        CudaBackendOptions options;
        options.max_batch = 300; // small, so batches split
        options.resident = resident;
        options.pinned = pinned;
        options.stage_timing = stage_timing;
        test_matches_cpu(base, queries, options);
      }

  test_failed_set_queries();

  // Enough round trips that the totals reach about 0.1 s, so one preemption
  // of the test thread cannot shift them much.
  for (bool resident : {false, true})
    for (bool stage_timing : {false, true}) {
      CudaBackendOptions options;
      options.max_batch = 1024;
      options.resident = resident;
      options.stage_timing = stage_timing;
      const CudaBackendStats s = run_batches(base, queries, options, 1000);
      check_times(s, stage_timing, 1000);
      // Gathering ids takes far less time than issuing the copies and kernel.
      if (resident && !stage_timing)
        CHECK(s.gather_us < 0.15 * s.submit_us);
    }

  // Large staged batches from pinned buffers, 8 MB of vectors at any
  // dimension: shipping the vectors takes longer than running the kernel or
  // copying the results back. Without the sync after each stage, to_device and
  // kernel would record only the time to issue them, and the wait would land
  // in a later stage.
  CudaBackendOptions large;
  large.max_batch = (8u << 20) / (base.d * sizeof(float));
  large.stage_timing = true;
  const CudaBackendStats s = run_batches(base, queries, large, 300);
  check_times(s, true, 300);
  CHECK(s.from_device_us < s.to_device_us + s.kernel_us);
  CHECK(s.to_device_us > s.kernel_us);

  // Without stage timing, waiting for the copies takes longer than issuing
  // them.
  large.stage_timing = false;
  const CudaBackendStats t = run_batches(base, queries, large, 300);
  check_times(t, false, 300);
  CHECK(t.wait_us > t.submit_us);

  CudaBackendOptions options;
  test_search_matches_reference(dir, base, queries, options);
  options.resident = true;
  test_search_matches_reference(dir, base, queries, options);

  CudaDistanceBackend gpu(base, options);
  CHECK(gpu.launch_floor_us(100) > 0.0);

  std::printf("test_cuda_backend passed\n");
  return 0;
}
