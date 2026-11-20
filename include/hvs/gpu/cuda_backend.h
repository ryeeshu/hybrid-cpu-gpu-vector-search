/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/backend.h"
#include "hvs/io.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hvs {

struct CudaBackendOptions {
  uint32_t max_batch = 4096; // candidates per round trip; larger batches split
  bool pinned = true;        // page-locked host staging buffers
  // false: gather candidate vectors on the host and ship them each batch.
  // true: keep the whole base on the device and ship only ids.
  bool resident = false;
  // Synchronize after each stage so its time can be measured. Adds overhead;
  // without it the time after gather is split only into submit and wait.
  bool stage_timing = false;
};

// Totals since construction or the last reset_stats(). After gather, a round
// trip is split into to_device, kernel and from_device with stage_timing, and
// into submit and wait without it; the other split's fields stay zero.
struct CudaBackendStats {
  uint64_t round_trips = 0;
  uint64_t candidates = 0;
  uint64_t bytes_to_device = 0;
  uint64_t bytes_from_device = 0;
  double gather_us = 0;     // host: filling the staging buffers
  double round_trip_us = 0; // gather through results back on the host
  double to_device_us = 0;  // these three only with stage_timing
  double kernel_us = 0;
  double from_device_us = 0;
  double submit_us = 0; // these two only without: issuing copies and kernel,
  double wait_us = 0;   // then blocking in the final synchronize
};

// Blocking CUDA distance backend with persistent buffers and its own stream.
// One instance per worker thread, as the DistanceBackend contract requires.
// Throws std::runtime_error on any CUDA failure.
class CudaDistanceBackend : public DistanceBackend {
public:
  CudaDistanceBackend(const FloatMatrix &base,
                      const CudaBackendOptions &options);
  ~CudaDistanceBackend() override;
  CudaDistanceBackend(const CudaDistanceBackend &) = delete;
  CudaDistanceBackend &operator=(const CudaDistanceBackend &) = delete;

  // Copies the queries to the device; the caller's array need not stay alive.
  void set_queries(const float *queries, uint32_t num_queries,
                   uint32_t dim) override;
  void compute_distances(const CandidateBatch &candidates,
                         std::vector<float> &distances) override;

  const CudaBackendStats &stats() const;
  void reset_stats();
  // Device memory this instance has allocated, in bytes.
  uint64_t device_bytes() const;
  // Mean time of launching an empty kernel and waiting for it, in
  // microseconds: the floor under any round trip.
  double launch_floor_us(uint32_t iterations);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The current device's name and the CUDA runtime and driver versions, such
// as "12.6".
struct CudaDeviceInfo {
  std::string name, runtime, driver;
};
CudaDeviceInfo cuda_device_info();

} // namespace hvs
