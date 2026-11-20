/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "hvs/backend.h"
#include "hvs/types.h"

#include <cstdio>
#include <vector>

using namespace hvs;

// Scalar reference backend, only to exercise the interface contract.
class ScalarBackend : public DistanceBackend {
public:
  ScalarBackend(const float *base, uint32_t dim) : base_(base), dim_(dim) {}

  void set_queries(const float *queries, uint32_t num_queries,
                   uint32_t dim) override {
    CHECK(dim == dim_);
    queries_ = queries;
    num_queries_ = num_queries;
  }

  void compute_distances(const CandidateBatch &candidates,
                         std::vector<float> &distances) override {
    distances.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      CHECK(candidates.query_ids[i] < num_queries_);
      const float *q = queries_ + size_t(candidates.query_ids[i]) * dim_;
      const float *v = base_ + size_t(candidates.vector_ids[i]) * dim_;
      float sum = 0.0f;
      for (uint32_t j = 0; j < dim_; ++j)
        sum += (q[j] - v[j]) * (q[j] - v[j]);
      distances[i] = sum;
    }
  }

private:
  const float *base_;
  const float *queries_ = nullptr;
  uint32_t num_queries_ = 0;
  uint32_t dim_;
};

int main() {
  CandidateBatch batch;
  CHECK(batch.empty());
  batch.push(0, 2);
  batch.push(1, 0);
  batch.push(1, 1);
  CHECK(batch.size() == 3);
  CHECK(batch.query_ids.size() == batch.vector_ids.size());

  const uint32_t dim = 2;
  const float base[] = {0, 0, 3, 4, 1, 1};
  const float queries[] = {0, 0, 1, 1};
  ScalarBackend backend(base, dim);
  backend.set_queries(queries, 2, dim);

  std::vector<float> distances;
  backend.compute_distances(batch, distances);
  CHECK(distances.size() == 3);
  CHECK(distances[0] == 2.0f);  // q0 to (1,1)
  CHECK(distances[1] == 2.0f);  // q1 to (0,0)
  CHECK(distances[2] == 13.0f); // q1 to (3,4)

  batch.clear();
  CHECK(batch.empty());
  backend.compute_distances(batch, distances);
  CHECK(distances.empty());

  // A cleared batch must not keep stale ids in either array.
  batch.push(0, 1);
  CHECK(batch.size() == 1 && batch.query_ids.size() == 1);
  backend.compute_distances(batch, distances);
  CHECK(distances.size() == 1 && distances[0] == 25.0f); // q0 to (3,4)

  SearchConfig config;
  CHECK(config.k <= config.ef_search);
  CHECK(INVALID_ID == 0xFFFFFFFFu);

  std::printf("test_types passed\n");
  return 0;
}
