/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include <cstdint>

#include <immintrin.h>

namespace hvs {

// Squared L2, one element at a time. Reference for the SIMD version.
inline float l2_scalar(const float *a, const float *b, uint32_t dim) {
  float sum = 0.0f;
  for (uint32_t j = 0; j < dim; ++j) {
    const float diff = a[j] - b[j];
    sum += diff * diff;
  }
  return sum;
}

// Squared L2 with AVX2 and FMA, 8 floats per step and a scalar tail. On
// integer-valued data every partial sum is exact, so it equals l2_scalar bit
// for bit; on other data it differs in the last bits.
inline float l2_avx2(const float *a, const float *b, uint32_t dim) {
  __m256 acc = _mm256_setzero_ps();
  uint32_t j = 0;
  for (; j + 8 <= dim; j += 8) {
    const __m256 diff =
        _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
    acc = _mm256_fmadd_ps(diff, diff, acc);
  }
  const __m128 half =
      _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
  const __m128 pair = _mm_add_ps(half, _mm_movehl_ps(half, half));
  float sum = _mm_cvtss_f32(_mm_add_ss(pair, _mm_shuffle_ps(pair, pair, 1)));
  for (; j < dim; ++j) {
    const float diff = a[j] - b[j];
    sum += diff * diff;
  }
  return sum;
}

inline float l2(const float *a, const float *b, uint32_t dim) {
  return l2_avx2(a, b, dim);
}

} // namespace hvs
