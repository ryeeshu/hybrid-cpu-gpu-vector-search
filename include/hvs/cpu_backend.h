/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/backend.h"
#include "hvs/dist.h"
#include "hvs/io.h"

#include <stdexcept>
#include <vector>

namespace hvs {

// Distance backend that computes on the calling thread. Used by tests and as
// the reference the CUDA backend is compared against.
class CpuDistanceBackend : public DistanceBackend {
public:
  explicit CpuDistanceBackend(const FloatMatrix &base) : base_(base) {}

  void set_queries(const float *queries, uint32_t num_queries,
                   uint32_t dim) override {
    if (dim != base_.d)
      throw std::invalid_argument("query dimension differs from the base");
    queries_ = queries;
    num_queries_ = num_queries;
  }

  void compute_distances(const CandidateBatch &candidates,
                         std::vector<float> &distances) override {
    if (candidates.query_ids.size() != candidates.vector_ids.size())
      throw std::invalid_argument("candidate arrays differ in length");
    distances.resize(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (candidates.query_ids[i] >= num_queries_ ||
          candidates.vector_ids[i] >= base_.n)
        throw std::out_of_range("candidate id out of range");
      distances[i] = l2(queries_ + size_t(candidates.query_ids[i]) * base_.d,
                        base_.row(candidates.vector_ids[i]), base_.d);
    }
    ++calls_;
    candidates_ += candidates.size();
  }

  // Completed compute_distances calls, empty ones included, and the
  // candidates they held, since construction or the last reset_counts().
  uint64_t calls() const { return calls_; }
  uint64_t candidates() const { return candidates_; }
  void reset_counts() {
    calls_ = 0;
    candidates_ = 0;
  }

private:
  const FloatMatrix &base_;
  const float *queries_ = nullptr;
  uint32_t num_queries_ = 0;
  uint64_t calls_ = 0;
  uint64_t candidates_ = 0;
};

} // namespace hvs
