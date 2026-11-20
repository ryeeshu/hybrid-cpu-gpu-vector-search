/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/backend.h"
#include "hvs/dist.h"
#include "hvs/io.h"
#include "hvs/types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <omp.h>

namespace hvs {

// (distance, id). Its built-in ordering is the total order used everywhere.
using Scored = std::pair<float, VectorId>;

// Open-addressed set of vector ids for one query. Grows when half full, so a
// lookup always finds an empty slot.
class VisitedSet {
public:
  // Empties the set and sizes it for about expected insertions.
  void reset(size_t expected) {
    size_t capacity = 1024;
    while (capacity < 2 * expected)
      capacity *= 2;
    slots_.assign(capacity, INVALID_ID);
    count_ = 0;
  }

  // Returns true if id was not in the set before.
  bool insert(VectorId id) {
    if (2 * (count_ + 1) > slots_.size())
      grow();
    return place(id);
  }

private:
  bool place(VectorId id) {
    const size_t mask = slots_.size() - 1;
    size_t i = (id * 0x9E3779B1u) & mask;
    while (slots_[i] != INVALID_ID) {
      if (slots_[i] == id)
        return false;
      i = (i + 1) & mask;
    }
    slots_[i] = id;
    ++count_;
    return true;
  }

  void grow() {
    std::vector<VectorId> old(slots_.size() * 2, INVALID_ID);
    old.swap(slots_);
    count_ = 0;
    for (VectorId id : old)
      if (id != INVALID_ID)
        place(id);
  }

  std::vector<VectorId> slots_;
  size_t count_ = 0;
};

// Level-0 best-first search for one query, split into steps so the distances
// can come from anywhere: next() names the vectors to evaluate, absorb() takes
// their distances. Each batch must be absorbed before the next call to next().
// Follows the rules documented with reference_search in
// tools/make_synthetic.py.
class QuerySearch {
public:
  QuerySearch(const Graph &graph, uint32_t ef, uint32_t k)
      : graph_(graph), ef_(ef), k_(k) {
    if (ef == 0)
      throw std::invalid_argument("ef must be positive");
  }

  // Starts a new query from the graph's entry point.
  void begin() {
    visited_.reset(size_t(ef_) * 32);
    candidates_.clear();
    results_.clear();
    entry_emitted_ = false;
    evals_ = 0;
    hops_ = 0;
    batches_ = 0;
  }

  // Fills ids with the next vectors whose distances are needed. Returns false
  // when the search is over.
  bool next(std::vector<VectorId> &ids) {
    ids.clear();
    if (!entry_emitted_) {
      entry_emitted_ = true;
      visited_.insert(graph_.entry_point);
      evals_ = 1;
      batches_ = 1;
      ids.push_back(graph_.entry_point);
      return true;
    }
    while (!candidates_.empty()) {
      std::pop_heap(candidates_.begin(), candidates_.end(), std::greater<>());
      const Scored nearest = candidates_.back();
      candidates_.pop_back();
      if (results_.size() == ef_ && nearest > results_.front())
        return false;
      ++hops_;
      const VectorId *row = graph_.row(nearest.second);
      for (uint32_t j = 0; j < graph_.degree; ++j)
        if (row[j] != INVALID_ID && visited_.insert(row[j]))
          ids.push_back(row[j]);
      evals_ += ids.size();
      if (!ids.empty()) { // an expansion with nothing new needs no distances
        ++batches_;
        return true;
      }
    }
    return false;
  }

  // Takes the distances for the ids returned by the last next().
  void absorb(const std::vector<VectorId> &ids, const float *distances) {
    for (size_t i = 0; i < ids.size(); ++i) {
      const Scored found(distances[i], ids[i]);
      if (results_.size() < ef_ || found < results_.front()) {
        candidates_.push_back(found);
        std::push_heap(candidates_.begin(), candidates_.end(),
                       std::greater<>());
        results_.push_back(found);
        std::push_heap(results_.begin(), results_.end());
        if (results_.size() > ef_) {
          std::pop_heap(results_.begin(), results_.end());
          results_.pop_back();
        }
      }
    }
  }

  // Writes the nearest min(k, ef) results, nearest first, and the counters.
  void finish(QueryId query_id, SearchResult &result) {
    std::sort(results_.begin(), results_.end());
    const size_t count = std::min<size_t>(k_, results_.size());
    result.query_id = query_id;
    result.ids.resize(count);
    result.distances.resize(count);
    for (size_t i = 0; i < count; ++i) {
      result.distances[i] = results_[i].first;
      result.ids[i] = results_[i].second;
    }
    result.distance_evals = evals_;
    result.hops = hops_;
    result.batches = batches_;
  }

private:
  const Graph &graph_;
  uint32_t ef_;
  uint32_t k_;
  VisitedSet visited_;
  std::vector<Scored> candidates_; // min-heap
  std::vector<Scored> results_;    // max-heap, at most ef entries
  bool entry_emitted_ = false;
  uint64_t evals_ = 0;
  uint64_t hops_ = 0;
  uint64_t batches_ = 0;
};

// V1 inner loop: one query to completion with inline SIMD distances. Each
// batch of vectors is prefetched before its distances are computed.
inline void search_cpu(QuerySearch &search, const FloatMatrix &base,
                       const float *query, QueryId query_id,
                       std::vector<VectorId> &ids,
                       std::vector<float> &distances, SearchResult &result) {
  search.begin();
  while (search.next(ids)) {
    for (VectorId id : ids)
      _mm_prefetch(reinterpret_cast<const char *>(base.row(id)), _MM_HINT_T0);
    distances.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i)
      distances[i] = l2(query, base.row(ids[i]), base.d);
    search.absorb(ids, distances.data());
  }
  search.finish(query_id, result);
}

// One query to completion with every distance requested from a backend, one
// blocking call per expansion. This is the per-query loop of the synchronous
// hybrid. The backend must already hold the queries.
inline void search_with_backend(QuerySearch &search, DistanceBackend &backend,
                                QueryId query_id, CandidateBatch &batch,
                                std::vector<float> &distances,
                                SearchResult &result) {
  search.begin();
  while (search.next(batch.vector_ids)) {
    batch.query_ids.assign(batch.vector_ids.size(), query_id);
    backend.compute_distances(batch, distances);
    search.absorb(batch.vector_ids, distances.data());
  }
  search.finish(query_id, result);
}

// Runs queries 0 to num_queries - 1 on a team of threads, the loop shared by
// every mode. Each thread calls make_worker(thread index) once, then calls the
// returned worker with each query id it is given. If latencies_us is given it
// receives each query's wall time in microseconds. The first exception thrown
// by any thread is rethrown after the region; queries not yet started are
// skipped. Returns the number of threads the region ran with.
template <typename MakeWorker>
uint32_t run_queries(uint32_t num_queries, uint32_t threads,
                     std::vector<double> *latencies_us,
                     MakeWorker &&make_worker) {
  if (threads == 0)
    throw std::invalid_argument("thread count must be positive");
  if (latencies_us)
    latencies_us->assign(num_queries, 0.0);
  using Worker = decltype(make_worker(uint32_t(0)));
  uint32_t team_size = 0;
  std::atomic<bool> failed(false);
  std::exception_ptr error;
  // Called from a catch block: keeps the first exception.
  const auto record_error = [&] {
#pragma omp critical(hvs_record_error)
    if (!error)
      error = std::current_exception();
    failed = true;
  };
#pragma omp parallel num_threads(threads)
  {
    if (omp_get_thread_num() == 0)
      team_size = uint32_t(omp_get_num_threads());
    std::optional<Worker> worker;
    try {
      worker.emplace(make_worker(uint32_t(omp_get_thread_num())));
    } catch (...) {
      record_error();
    }
#pragma omp for schedule(dynamic, 16)
    for (int64_t q = 0; q < int64_t(num_queries); ++q) {
      if (!worker || failed.load(std::memory_order_relaxed))
        continue;
      try {
        const auto start = std::chrono::steady_clock::now();
        (*worker)(QueryId(q));
        if (latencies_us)
          (*latencies_us)[q] = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
      } catch (...) {
        record_error();
      }
    }
  }
  if (error)
    std::rethrow_exception(error);
  return team_size;
}

// V1: every query in queries, config.cpu_threads at a time. If latencies_us
// is given it receives each query's wall time in microseconds; if
// threads_used is given it receives the number of threads that ran.
inline std::vector<SearchResult>
search_all_cpu(const Graph &graph, const FloatMatrix &base,
               const FloatMatrix &queries, const SearchConfig &config,
               std::vector<double> *latencies_us = nullptr,
               uint32_t *threads_used = nullptr) {
  std::vector<SearchResult> results(queries.n);
  const uint32_t team_size =
      run_queries(queries.n, config.cpu_threads, latencies_us, [&](uint32_t) {
        return [&, search = QuerySearch(graph, config.ef_search, config.k),
                ids = std::vector<VectorId>(),
                distances = std::vector<float>()](QueryId q) mutable {
          search_cpu(search, base, queries.row(q), q, ids, distances,
                     results[q]);
        };
      });
  if (threads_used)
    *threads_used = team_size;
  return results;
}

// V2, the synchronous hybrid: like search_all_cpu, but every distance comes
// from a backend in one blocking call per expansion. Thread t uses only
// backends[t]. The caller creates the backends, one per thread, and gives
// each the queries before the call.
inline std::vector<SearchResult>
search_all_hybrid_sync(const Graph &graph, const FloatMatrix &queries,
                       const SearchConfig &config,
                       const std::vector<DistanceBackend *> &backends,
                       std::vector<double> *latencies_us = nullptr,
                       uint32_t *threads_used = nullptr) {
  if (backends.size() != config.cpu_threads)
    throw std::invalid_argument("need exactly one backend per thread");
  for (const DistanceBackend *backend : backends)
    if (!backend)
      throw std::invalid_argument("backend is null");
  std::vector<SearchResult> results(queries.n);
  const uint32_t team_size = run_queries(
      queries.n, config.cpu_threads, latencies_us, [&](uint32_t thread) {
        return [&, &backend = *backends.at(thread),
                search = QuerySearch(graph, config.ef_search, config.k),
                batch = CandidateBatch(),
                distances = std::vector<float>()](QueryId q) mutable {
          search_with_backend(search, backend, q, batch, distances, results[q]);
        };
      });
  if (threads_used)
    *threads_used = team_size;
  return results;
}

} // namespace hvs
