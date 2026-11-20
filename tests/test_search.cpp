/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "hvs/cpu_backend.h"
#include "hvs/dist.h"
#include "hvs/io.h"
#include "hvs/search.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace hvs;

struct Dataset {
  FloatMatrix base, queries;
  Graph graph;
};

// A result must match the reference in ids, distance evaluations and hops.
static void check_against(const std::vector<SearchResult> &found,
                          const IdMatrix &ref_ids, const IdMatrix &ref_work) {
  CHECK(found.size() == ref_ids.n);
  for (uint32_t q = 0; q < ref_ids.n; ++q) {
    const SearchResult &r = found[q];
    CHECK(r.query_id == q);
    CHECK(r.ids.size() == ref_ids.d && r.distances.size() == ref_ids.d);
    for (uint32_t i = 0; i < ref_ids.d; ++i)
      CHECK(r.ids[i] == ref_ids.row(q)[i]);
    CHECK(r.distance_evals == ref_work.row(q)[0]);
    CHECK(r.hops == ref_work.row(q)[1]);
  }
}

// Results are nearest first under (distance, id), with true distances.
static void check_well_formed(const std::vector<SearchResult> &found,
                              const Dataset &data, size_t expected_size) {
  for (const SearchResult &r : found) {
    CHECK(r.ids.size() == expected_size);
    for (size_t i = 0; i < r.ids.size(); ++i) {
      CHECK(r.ids[i] < data.base.n);
      CHECK(r.distances[i] == l2_scalar(data.queries.row(r.query_id),
                                        data.base.row(r.ids[i]), data.base.d));
      if (i > 0)
        CHECK(Scored(r.distances[i - 1], r.ids[i - 1]) <
              Scored(r.distances[i], r.ids[i]));
    }
  }
}

static std::vector<SearchResult> run_with_backend(const Dataset &data,
                                                  const SearchConfig &config) {
  CpuDistanceBackend backend(data.base);
  backend.set_queries(data.queries.data.data(), data.queries.n, data.queries.d);
  QuerySearch search(data.graph, config.ef_search, config.k);
  CandidateBatch batch;
  std::vector<float> distances;
  std::vector<SearchResult> results(data.queries.n);
  for (uint32_t q = 0; q < data.queries.n; ++q)
    search_with_backend(search, backend, q, batch, distances, results[q]);
  return results;
}

static void test_distance_kernels() {
  // Dimensions around the 8-wide step, integer-valued so sums are exact.
  for (uint32_t dim : {1u, 6u, 7u, 8u, 9u, 33u, 128u}) {
    std::vector<float> a(dim), b(dim);
    for (uint32_t j = 0; j < dim; ++j) {
      a[j] = float((j * 37 + 11) % 256);
      b[j] = float((j * 101 + 5) % 256);
    }
    CHECK(l2_avx2(a.data(), b.data(), dim) ==
          l2_scalar(a.data(), b.data(), dim));
  }
}

static void test_visited_set() {
  VisitedSet set;
  set.reset(4);
  for (VectorId id = 0; id < 5000; ++id) // forces several growth steps
    CHECK(set.insert(id * 7919));
  for (VectorId id = 0; id < 5000; ++id)
    CHECK(!set.insert(id * 7919));
  set.reset(4);
  CHECK(set.insert(0));
}

int main(int argc, char **argv) {
  CHECK(argc == 2); // synthetic data directory
  const std::string dir = argv[1];
  test_distance_kernels();
  test_visited_set();

  Dataset data{load_matrix<float>(dir + "/base.fbin"),
               load_matrix<float>(dir + "/query.fbin"),
               load_graph(dir + "/graph0.bin", dir + "/upper.bin")};

  for (uint32_t ef : {10u, 32u, 64u}) {
    const std::string stem = dir + "/ref_ef" + std::to_string(ef);
    const IdMatrix ref_ids = load_matrix<VectorId>(stem + ".ibin");
    const IdMatrix ref_work = load_matrix<VectorId>(stem + "_work.ibin");
    SearchConfig config;
    config.k = ref_ids.d;
    config.ef_search = ef;

    // The same answer whether distances are inline or come from a backend,
    // and at any thread count.
    for (uint32_t threads : {1u, 4u}) {
      config.cpu_threads = threads;
      const auto found =
          search_all_cpu(data.graph, data.base, data.queries, config);
      check_against(found, ref_ids, ref_work);
      check_well_formed(found, data, ref_ids.d);
    }
    check_against(run_with_backend(data, config), ref_ids, ref_work);
  }

  // With ef below k the result holds ef entries.
  SearchConfig small;
  small.k = 10;
  small.ef_search = 5;
  check_well_formed(search_all_cpu(data.graph, data.base, data.queries, small),
                    data, 5);

  std::printf("test_search passed\n");
  return 0;
}
