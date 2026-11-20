/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hvs {

using VectorId = uint32_t;
using QueryId = uint32_t;

constexpr VectorId INVALID_ID = 0xFFFFFFFFu;

// Distance requests as parallel arrays: pair i asks for the distance between
// query query_ids[i] and database vector vector_ids[i]. Kept as arrays so the
// id lists can be copied to the GPU without repacking.
struct CandidateBatch {
  std::vector<QueryId> query_ids;
  std::vector<VectorId> vector_ids;

  size_t size() const { return vector_ids.size(); }
  bool empty() const { return vector_ids.empty(); }

  void clear() {
    query_ids.clear();
    vector_ids.clear();
  }

  void push(QueryId query_id, VectorId vector_id) {
    query_ids.push_back(query_id);
    vector_ids.push_back(vector_id);
  }
};

enum class ExecutionMode { CPU, HYBRID_SYNC, HYBRID_BATCH, HYBRID_PIPELINE };

struct SearchConfig {
  uint32_t k = 10;
  uint32_t ef_search = 64;
  uint32_t cpu_threads = 1;
  uint32_t batch_size = 4096; // max candidates per GPU batch
  // Bytes of engine-managed device memory.
  uint64_t gpu_memory_budget = uint64_t{256} << 20;
  uint32_t streams = 1;
  ExecutionMode mode = ExecutionMode::CPU;
};

// Top-k of one query, nearest first, ordered by (distance, id).
struct SearchResult {
  QueryId query_id = 0;
  std::vector<VectorId> ids;
  std::vector<float> distances;
  uint64_t distance_evals = 0;
  uint64_t hops = 0;
  // Distance batches requested: one for the entry point plus one per
  // expansion that found at least one new vector.
  uint64_t batches = 0;
};

} // namespace hvs
