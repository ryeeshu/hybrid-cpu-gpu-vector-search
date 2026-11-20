/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

// Prices one distance round trip against batch size, on a real dataset with
// random candidate ids: shipping vectors (pageable and pinned), shipping ids
// to a device-resident base, the host gather alone, and the CPU computing the
// same distances itself. Fresh ids are used every iteration so the host does
// not serve repeats from cache. Every time is the median over the iterations
// at one batch size, from one calling thread.

#include "options.h"
#include "report.h"
#include "runner.h"

#include "hvs/dist.h"
#include "hvs/gpu/cuda_backend.h"
#include "hvs/io.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

using namespace hvs;

// Each configuration runs at least this long and this often before it is
// timed.
constexpr double WARMUP_US = 50000;
constexpr uint32_t WARMUP_MIN = 3;

// The columns of xfer.csv; tools/plot.py reads them by name.
const std::string XFER_HEADER =
    "build_id,tag,host,platform,compiler,build_type,gpu,logical_cpus,"
    "cpu_threads,omp_proc_bind,omp_places,dataset_hash,n,d,run,pairs,"
    "iterations,launch_floor_us,ship_pageable_us,ship_pinned_us,gather_us,"
    "staged_round_trip_us,to_device_us,kernel_us,from_device_us,resident_us,"
    "cpu_1t_us,cpu_all_us,start_time_utc\n";

struct IdPool {
  std::vector<QueryId> query_ids;
  std::vector<VectorId> vector_ids;
  size_t cursor = 0;

  // Fills batch with the next count pairs, wrapping around the pool.
  void take(uint32_t count, CandidateBatch &batch) {
    if (cursor + count > vector_ids.size())
      cursor = 0;
    batch.query_ids.assign(query_ids.begin() + cursor,
                           query_ids.begin() + cursor + count);
    batch.vector_ids.assign(vector_ids.begin() + cursor,
                            vector_ids.begin() + cursor + count);
    cursor += count;
  }
};

static uint32_t iterations_for(uint32_t count) {
  return std::clamp<uint32_t>(4000000u / count, 8u, 2000u);
}

static double elapsed_us(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now() - since)
      .count();
}

// Runs step until it has run WARMUP_MIN times and for WARMUP_US.
template <typename Step> static void warm_up(Step step) {
  const auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < WARMUP_MIN || elapsed_us(start) < WARMUP_US; ++i)
    step();
}

// The lower middle value for an even count.
static double median(const std::vector<double> &sample) {
  return percentile(sample, 50);
}

// Median per-round-trip statistics of a backend configuration at one batch
// size. The backend keeps totals, so one iteration is the difference of its
// statistics after and before one round trip.
static CudaBackendStats measure_gpu(const FloatMatrix &base,
                                    const FloatMatrix &queries, IdPool &pool,
                                    uint32_t count,
                                    CudaBackendOptions options) {
  options.max_batch = count;
  CudaDistanceBackend gpu(base, options);
  gpu.set_queries(queries.data.data(), queries.n, queries.d);
  CandidateBatch batch;
  std::vector<float> distances;
  const auto round_trip = [&] {
    pool.take(count, batch);
    gpu.compute_distances(batch, distances);
  };
  warm_up(round_trip);
  std::vector<double> total, gather, to_device, kernel, from_device;
  for (uint32_t i = 0; i < iterations_for(count); ++i) {
    const CudaBackendStats before = gpu.stats();
    round_trip();
    const CudaBackendStats &after = gpu.stats();
    total.push_back(after.round_trip_us - before.round_trip_us);
    gather.push_back(after.gather_us - before.gather_us);
    to_device.push_back(after.to_device_us - before.to_device_us);
    kernel.push_back(after.kernel_us - before.kernel_us);
    from_device.push_back(after.from_device_us - before.from_device_us);
  }
  CudaBackendStats medians;
  medians.round_trip_us = median(total);
  medians.gather_us = median(gather);
  medians.to_device_us = median(to_device);
  medians.kernel_us = median(kernel);
  medians.from_device_us = median(from_device);
  return medians;
}

// Median microseconds for the CPU to compute count distances with threads.
// Adds the distances to checksum. Throws unless the loop ran on threads.
static double measure_cpu(const FloatMatrix &base, const FloatMatrix &queries,
                          IdPool &pool, uint32_t count, int threads,
                          double &checksum) {
  CandidateBatch batch;
  std::vector<float> distances(count);
  int team = 0;
  const auto pass = [&] {
    pool.take(count, batch);
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel num_threads(threads) if (threads > 1)
    {
      if (omp_get_thread_num() == 0)
        team = omp_get_num_threads();
#pragma omp for schedule(static)
      for (int64_t i = 0; i < int64_t(count); ++i)
        distances[i] = l2(queries.row(batch.query_ids[i]),
                          base.row(batch.vector_ids[i]), base.d);
    }
    const double us = elapsed_us(start);
    // Read after timing, so the compiler cannot drop the distance loop.
    checksum += std::accumulate(distances.begin(), distances.end(), 0.0);
    return us;
  };
  warm_up(pass);
  std::vector<double> times;
  for (uint32_t i = 0; i < iterations_for(count); ++i)
    times.push_back(pass());
  if (team != threads)
    throw std::runtime_error("asked for " + std::to_string(threads) +
                             " threads but ran on " + std::to_string(team) +
                             "; check OMP_THREAD_LIMIT and OMP_DYNAMIC");
  return median(times);
}

int main(int argc, char **argv) {
  std::string data, csv_path, tag = "untagged";
  uint32_t run = 1, max_mb = 256;
  try {
    for (int i = 1; i < argc; i += 2) {
      const std::string flag = argv[i];
      if (i + 1 >= argc)
        throw std::invalid_argument("missing value for " + flag);
      const std::string value = argv[i + 1];
      if (flag == "--data")
        data = value;
      else if (flag == "--csv")
        csv_path = value;
      else if (flag == "--tag")
        tag = value;
      else if (flag == "--run")
        run = parse_u32(flag, value);
      else if (flag == "--max-mb")
        max_mb = parse_u32(flag, value);
      else
        throw std::invalid_argument("unknown flag " + flag);
    }
    if (data.empty())
      throw std::invalid_argument("--data is required");
    if (max_mb == 0)
      throw std::invalid_argument("--max-mb must be positive");
    // The tag is written into the CSV unquoted.
    if (tag.empty() || tag.find_first_of(",\"'\r\n") != tag.npos)
      throw std::invalid_argument(
          "--tag must be non-empty, without commas, quotes or newlines");
    if (!csv_path.empty())
      check_csv(csv_path, XFER_HEADER); // before anything is measured

    // Loaded as hvs_search loads it, so the provenance columns match.
    Options o;
    o.data = data;
    o.tag = tag;
    const Dataset dataset = load_dataset(o);
    const ReportRow launch = launch_row(o, dataset);
    const FloatMatrix &base = dataset.base, &queries = dataset.queries;

    IdPool pool;
    std::mt19937 rng(1);
    pool.query_ids.resize(8u << 20);
    pool.vector_ids.resize(8u << 20);
    for (size_t i = 0; i < pool.vector_ids.size(); ++i) {
      pool.query_ids[i] = rng() % queries.n;
      pool.vector_ids[i] = rng() % base.n;
    }

    const std::string gpu_name = csv_safe(cuda_device_info().name);
    const int all_threads = omp_get_max_threads();
    // Each call times one empty kernel, after an untimed one.
    CudaBackendOptions probe_options;
    probe_options.max_batch = 1;
    CudaDistanceBackend probe(base, probe_options);
    warm_up([&] { probe.launch_floor_us(1); });
    std::vector<double> floors;
    for (int i = 0; i < 2000; ++i)
      floors.push_back(probe.launch_floor_us(1));
    const double floor_us = median(floors);

    std::printf("%s  %s  n=%u d=%u  run %u  empty kernel launch + wait: "
                "%.1f us\n",
                launch.host.c_str(), gpu_name.c_str(), base.n, base.d, run,
                floor_us);
    std::printf("median microseconds per round trip, one calling thread; "
                "ship = vectors cross the bus, resident = ids only\n");
    std::printf("ship page is a single-caller figure (slower with concurrent "
                "callers); cpu all = %d threads\n",
                all_threads);
    std::printf("%8s %10s %10s %9s %10s %8s %8s %8s %10s %10s %10s\n", "pairs",
                "ship page", "ship pin", "(gather)", "staged rt", "to dev",
                "kernel", "from dev", "resident", "cpu 1t", "cpu all");

    // The columns that are the same on every row, through run.
    const std::string prefix =
        launch.build_id + "," + launch.tag + "," + launch.host + "," +
        launch.platform + "," + launch.compiler + "," + launch.build_type +
        "," + gpu_name + "," + std::to_string(launch.logical_cpus) + "," +
        std::to_string(all_threads) + "," + launch.omp_proc_bind + "," +
        launch.omp_places + "," + launch.dataset_hash + "," +
        std::to_string(launch.n) + "," + std::to_string(launch.d) + "," +
        std::to_string(run) + ",";
    std::vector<std::string> lines;
    double checksum = 0;

    // 1 to 64 by doubling, then by powers of 4, up to max_mb of vectors.
    const uint64_t max_bytes = uint64_t(max_mb) << 20;
    for (uint64_t count = 1; count * base.d * sizeof(float) <= max_bytes &&
                             count <= pool.vector_ids.size();
         count *= (count < 64 ? 2 : 4)) {
      const uint32_t pairs = uint32_t(count);
      CudaBackendOptions pageable, pinned, staged, resident;
      pageable.pinned = false;
      staged.stage_timing = true;
      resident.resident = true;
      const double page_us =
          measure_gpu(base, queries, pool, pairs, pageable).round_trip_us;
      const CudaBackendStats pin =
          measure_gpu(base, queries, pool, pairs, pinned);
      const CudaBackendStats st =
          measure_gpu(base, queries, pool, pairs, staged);
      const double res_us =
          measure_gpu(base, queries, pool, pairs, resident).round_trip_us;
      const double cpu1 = measure_cpu(base, queries, pool, pairs, 1, checksum);
      const double cpu_all =
          measure_cpu(base, queries, pool, pairs, all_threads, checksum);
      std::printf("%8u %10.1f %10.1f %9.1f %10.1f %8.1f %8.1f %8.1f %10.1f "
                  "%10.1f %10.1f\n",
                  pairs, page_us, pin.round_trip_us, pin.gather_us,
                  st.round_trip_us, st.to_device_us, st.kernel_us,
                  st.from_device_us, res_us, cpu1, cpu_all);
      char numbers[512];
      std::snprintf(numbers, sizeof(numbers),
                    "%u,%u,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,"
                    "%.2f,",
                    pairs, iterations_for(pairs), floor_us, page_us,
                    pin.round_trip_us, pin.gather_us, st.round_trip_us,
                    st.to_device_us, st.kernel_us, st.from_device_us, res_us,
                    cpu1, cpu_all);
      lines.push_back(prefix + numbers + launch.start_time_utc + "\n");
    }
    // Printed so the CPU distances are used.
    std::printf("checksum of the CPU distances: %.6e\n", checksum);
    if (!csv_path.empty())
      append_csv(csv_path, XFER_HEADER, lines);
  } catch (const std::invalid_argument &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    std::fprintf(stderr, "usage: xfer_bench --data DIR [--max-mb N] "
                         "[--csv FILE] [--tag TEXT] [--run N]\n");
    return 1;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
