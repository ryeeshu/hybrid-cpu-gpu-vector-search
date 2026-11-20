/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "hvs/gpu/cuda_backend.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// Throws with the CUDA error string if a runtime call fails. The thread's last
// error is cleared first, so a later cudaGetLastError() does not report the
// same failure again.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t cuda_err = (call);                                       \
    if (cuda_err != cudaSuccess) {                                             \
      (void)cudaGetLastError();                                                \
      throw std::runtime_error(std::string(#call) +                            \
                               " failed: " + cudaGetErrorString(cuda_err));    \
    }                                                                          \
  } while (0)

namespace {

constexpr int THREADS_PER_BLOCK = 256;

// One thread per pair; vectors arrive staged, pair i at row i.
__global__ void staged_kernel(const float *vectors, const float *queries,
                              const uint32_t *query_ids, uint32_t count,
                              uint32_t dim, float *out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float *v = vectors + size_t(i) * dim;
  const float *q = queries + size_t(query_ids[i]) * dim;
  float sum = 0.0f;
  for (uint32_t j = 0; j < dim; ++j) {
    const float diff = q[j] - v[j];
    sum += diff * diff;
  }
  out[i] = sum;
}

// One thread per pair; vectors are read from the device-resident base by id.
__global__ void resident_kernel(const float *base, const uint32_t *vector_ids,
                                const float *queries, const uint32_t *query_ids,
                                uint32_t count, uint32_t dim, float *out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const float *v = base + size_t(vector_ids[i]) * dim;
  const float *q = queries + size_t(query_ids[i]) * dim;
  float sum = 0.0f;
  for (uint32_t j = 0; j < dim; ++j) {
    const float diff = q[j] - v[j];
    sum += diff * diff;
  }
  out[i] = sum;
}

__global__ void empty_kernel() {}

double elapsed_us(std::chrono::steady_clock::time_point since,
                  std::chrono::steady_clock::time_point until =
                      std::chrono::steady_clock::now()) {
  return std::chrono::duration<double, std::micro>(until - since).count();
}

} // namespace

namespace hvs {

struct CudaDistanceBackend::Impl {
  const FloatMatrix &base;
  CudaBackendOptions options;
  CudaBackendStats stats;
  cudaStream_t stream = nullptr;
  uint64_t device_bytes = 0;
  uint64_t query_bytes = 0; // size of d_queries
  uint32_t num_queries = 0;

  // Host staging, capacity max_batch.
  float *h_vectors = nullptr;
  uint32_t *h_vector_ids = nullptr;
  uint32_t *h_query_ids = nullptr;
  float *h_out = nullptr;

  float *d_base = nullptr;
  float *d_queries = nullptr;
  float *d_vectors = nullptr;
  uint32_t *d_vector_ids = nullptr;
  uint32_t *d_query_ids = nullptr;
  float *d_out = nullptr;

  Impl(const FloatMatrix &base_vectors, const CudaBackendOptions &opts)
      : base(base_vectors), options(opts) {}

  template <typename T> void host_alloc(T *&ptr, size_t count) {
    if (options.pinned)
      CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    else if (!(ptr = static_cast<T *>(std::malloc(count * sizeof(T)))))
      throw std::bad_alloc();
  }

  template <typename T> void host_free(T *ptr) {
    if (options.pinned)
      cudaFreeHost(ptr);
    else
      std::free(ptr);
  }

  template <typename T> void device_alloc(T *&ptr, size_t count) {
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    device_bytes += count * sizeof(T);
  }

  void release() {
    host_free(h_vectors);
    host_free(h_vector_ids);
    host_free(h_query_ids);
    host_free(h_out);
    cudaFree(d_base);
    cudaFree(d_queries);
    cudaFree(d_vectors);
    cudaFree(d_vector_ids);
    cudaFree(d_query_ids);
    cudaFree(d_out);
    if (stream)
      cudaStreamDestroy(stream);
  }

  // One round trip for count <= max_batch pairs.
  void round_trip(const QueryId *query_ids, const VectorId *vector_ids,
                  uint32_t count, float *out) {
    const uint32_t dim = base.d;
    const auto start = std::chrono::steady_clock::now();
    std::memcpy(h_query_ids, query_ids, count * sizeof(uint32_t));
    if (options.resident)
      std::memcpy(h_vector_ids, vector_ids, count * sizeof(uint32_t));
    else
      for (uint32_t i = 0; i < count; ++i)
        std::memcpy(h_vectors + size_t(i) * dim, base.row(vector_ids[i]),
                    dim * sizeof(float));
    stats.gather_us += elapsed_us(start);

    auto stage = std::chrono::steady_clock::now();
    const size_t payload = options.resident
                               ? count * sizeof(uint32_t)
                               : size_t(count) * dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_query_ids, h_query_ids,
                               count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                               stream));
    if (options.resident)
      CUDA_CHECK(cudaMemcpyAsync(d_vector_ids, h_vector_ids, payload,
                                 cudaMemcpyHostToDevice, stream));
    else
      CUDA_CHECK(cudaMemcpyAsync(d_vectors, h_vectors, payload,
                                 cudaMemcpyHostToDevice, stream));
    if (options.stage_timing) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      stats.to_device_us += elapsed_us(stage);
      stage = std::chrono::steady_clock::now();
    }

    const uint32_t blocks = (count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (options.resident)
      resident_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          d_base, d_vector_ids, d_queries, d_query_ids, count, dim, d_out);
    else
      staged_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
          d_vectors, d_queries, d_query_ids, count, dim, d_out);
    CUDA_CHECK(cudaGetLastError());
    if (options.stage_timing) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      stats.kernel_us += elapsed_us(stage);
      stage = std::chrono::steady_clock::now();
    }

    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    const auto wait = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (options.stage_timing) {
      stats.from_device_us += elapsed_us(stage);
    } else {
      stats.submit_us += elapsed_us(stage, wait);
      stats.wait_us += elapsed_us(wait);
    }
    std::memcpy(out, h_out, count * sizeof(float));

    stats.round_trip_us += elapsed_us(start);
    stats.round_trips += 1;
    stats.candidates += count;
    stats.bytes_to_device += payload + count * sizeof(uint32_t);
    stats.bytes_from_device += count * sizeof(float);
  }
};

CudaDistanceBackend::CudaDistanceBackend(const FloatMatrix &base,
                                         const CudaBackendOptions &options)
    : impl_(new Impl(base, options)) {
  if (options.max_batch == 0)
    throw std::invalid_argument("max_batch must be positive");
  const size_t batch = options.max_batch;
  try {
    CUDA_CHECK(cudaStreamCreate(&impl_->stream));
    impl_->host_alloc(impl_->h_query_ids, batch);
    impl_->host_alloc(impl_->h_out, batch);
    impl_->device_alloc(impl_->d_query_ids, batch);
    impl_->device_alloc(impl_->d_out, batch);
    if (options.resident) {
      impl_->host_alloc(impl_->h_vector_ids, batch);
      impl_->device_alloc(impl_->d_vector_ids, batch);
      impl_->device_alloc(impl_->d_base, base.data.size());
      CUDA_CHECK(cudaMemcpy(impl_->d_base, base.data.data(),
                            base.data.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
    } else {
      impl_->host_alloc(impl_->h_vectors, batch * base.d);
      impl_->device_alloc(impl_->d_vectors, batch * base.d);
    }
  } catch (...) {
    impl_->release();
    throw;
  }
}

CudaDistanceBackend::~CudaDistanceBackend() { impl_->release(); }

void CudaDistanceBackend::set_queries(const float *queries,
                                      uint32_t num_queries, uint32_t dim) {
  if (dim != impl_->base.d)
    throw std::invalid_argument("query dimension differs from the base");
  // Drop the old queries first, so a failure below leaves none registered.
  impl_->num_queries = 0;
  if (impl_->d_queries) {
    CUDA_CHECK(cudaFree(impl_->d_queries));
    impl_->device_bytes -= impl_->query_bytes;
    impl_->d_queries = nullptr;
  }
  const size_t count = size_t(num_queries) * dim;
  impl_->device_alloc(impl_->d_queries, count);
  impl_->query_bytes = count * sizeof(float);
  CUDA_CHECK(cudaMemcpy(impl_->d_queries, queries, count * sizeof(float),
                        cudaMemcpyHostToDevice));
  impl_->num_queries = num_queries;
}

void CudaDistanceBackend::compute_distances(const CandidateBatch &candidates,
                                            std::vector<float> &distances) {
  if (candidates.query_ids.size() != candidates.vector_ids.size())
    throw std::invalid_argument("candidate arrays differ in length");
  // An out-of-range id would read outside a device buffer, so this stays on.
  for (size_t i = 0; i < candidates.size(); ++i)
    if (candidates.query_ids[i] >= impl_->num_queries ||
        candidates.vector_ids[i] >= impl_->base.n)
      throw std::out_of_range("candidate id out of range");

  distances.resize(candidates.size());
  for (size_t done = 0; done < candidates.size();
       done += impl_->options.max_batch) {
    const uint32_t count = uint32_t(
        std::min<size_t>(impl_->options.max_batch, candidates.size() - done));
    impl_->round_trip(candidates.query_ids.data() + done,
                      candidates.vector_ids.data() + done, count,
                      distances.data() + done);
  }
}

const CudaBackendStats &CudaDistanceBackend::stats() const {
  return impl_->stats;
}

void CudaDistanceBackend::reset_stats() { impl_->stats = CudaBackendStats(); }

uint64_t CudaDistanceBackend::device_bytes() const {
  return impl_->device_bytes;
}

double CudaDistanceBackend::launch_floor_us(uint32_t iterations) {
  empty_kernel<<<1, 1, 0, impl_->stream>>>(); // first launch pays setup costs
  CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  const auto start = std::chrono::steady_clock::now();
  for (uint32_t i = 0; i < iterations; ++i) {
    empty_kernel<<<1, 1, 0, impl_->stream>>>();
    CUDA_CHECK(cudaStreamSynchronize(impl_->stream));
  }
  CUDA_CHECK(cudaGetLastError());
  return elapsed_us(start) / iterations;
}

// A CUDA version number such as 12060 as "12.6".
static std::string version_text(int version) {
  return std::to_string(version / 1000) + "." +
         std::to_string(version % 1000 / 10);
}

CudaDeviceInfo cuda_device_info() {
  int device = 0, runtime = 0, driver = 0;
  cudaDeviceProp props;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&props, device));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CudaDeviceInfo info;
  info.name = props.name;
  info.runtime = version_text(runtime);
  info.driver = version_text(driver);
  return info;
}

} // namespace hvs
