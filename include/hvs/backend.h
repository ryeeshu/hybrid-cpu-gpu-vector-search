/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/types.h"

#include <cstdint>
#include <vector>

namespace hvs {

// Computes squared L2 distances for candidate pairs. Traversal owns the search
// state; a backend never touches a query's frontier.
//
// Contract:
// - A backend instance is not thread-safe. Each worker thread uses its own
//   instance; instances may share read-only data such as the base vectors.
// - The base vectors are given to an implementation at construction and must
//   outlive it.
class DistanceBackend {
public:
  virtual ~DistanceBackend() = default;

  // Registers num_queries row-major query vectors of dimension dim. A
  // QueryId in a later batch indexes into this array. The backend may keep
  // the pointer instead of copying, so the array must stay valid and
  // unchanged until the next set_queries call or the backend's destruction.
  virtual void set_queries(const float *queries, uint32_t num_queries,
                           uint32_t dim) = 0;

  // Resizes distances to candidates.size() and sets distances[i] for pair i.
  // Blocks until every distance is available.
  virtual void compute_distances(const CandidateBatch &candidates,
                                 std::vector<float> &distances) = 0;
};

} // namespace hvs
