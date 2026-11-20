/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include "hvs/types.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hvs {

// Row-major n x d matrix. Files are a (uint32 n, uint32 d) header followed by
// n * d little-endian values.
template <typename T> struct Matrix {
  uint32_t n = 0;
  uint32_t d = 0;
  std::vector<T> data;

  const T *row(size_t i) const { return data.data() + i * d; }
};

using FloatMatrix = Matrix<float>;
using IdMatrix = Matrix<VectorId>;

// One upper HNSW level: the nodes present on it (sorted) and their lists.
struct UpperLevel {
  uint32_t degree = 0;
  std::vector<VectorId> nodes;
  std::vector<VectorId> neighbors; // nodes.size() x degree
};

// Level-0 adjacency with fixed degree; unused slots hold INVALID_ID and may
// appear anywhere in a row.
struct Graph {
  uint32_t n = 0;
  uint32_t degree = 0;
  VectorId entry_point = 0;
  std::vector<VectorId> neighbors;
  std::vector<UpperLevel> upper;

  const VectorId *row(size_t i) const { return neighbors.data() + i * degree; }
};

namespace detail {

inline void fail(const std::string &path, const std::string &what) {
  throw std::runtime_error(path + ": " + what);
}

template <typename T>
void read_values(std::ifstream &in, const std::string &path, T *out,
                 size_t count) {
  in.read(reinterpret_cast<char *>(out), std::streamsize(count * sizeof(T)));
  if (size_t(in.gcount()) != count * sizeof(T))
    fail(path, "truncated");
}

inline void check_ids(const std::string &path, const std::vector<VectorId> &ids,
                      uint32_t n, bool allow_invalid) {
  for (VectorId id : ids)
    if (id >= n && !(allow_invalid && id == INVALID_ID))
      fail(path, "id " + std::to_string(id) + " out of range");
}

// True if the sorted list holds id.
inline bool contains(const std::vector<VectorId> &sorted, VectorId id) {
  return std::binary_search(sorted.begin(), sorted.end(), id);
}

} // namespace detail

// Loads an n x d matrix and checks that the file holds exactly that much.
template <typename T> Matrix<T> load_matrix(const std::string &path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in)
    detail::fail(path, "cannot open");
  const uint64_t file_size = uint64_t(in.tellg());
  in.seekg(0);

  Matrix<T> m;
  uint32_t header[2];
  detail::read_values(in, path, header, 2);
  m.n = header[0];
  m.d = header[1];
  // Compared by division so a huge header cannot wrap the product.
  const uint64_t values = (file_size - 8) / sizeof(T);
  if (m.n == 0 || m.d == 0 || (file_size - 8) % sizeof(T) != 0 ||
      values % m.d != 0 || values / m.d != m.n)
    detail::fail(path, "size " + std::to_string(file_size) +
                           " does not match " + std::to_string(m.n) + " x " +
                           std::to_string(m.d));
  m.data.resize(size_t(m.n) * m.d);
  detail::read_values(in, path, m.data.data(), m.data.size());
  return m;
}

// Loads graph0.bin and upper.bin. Every id is validated once here, so
// traversal needs no per-step bounds checks. Upper levels are checked for the
// structure a descent relies on: nodes sorted and unique, each level a subset
// of the one below, neighbors on their own level, entry point on the top.
inline Graph load_graph(const std::string &graph0_path,
                        const std::string &upper_path) {
  IdMatrix level0 = load_matrix<VectorId>(graph0_path);
  Graph g;
  g.n = level0.n;
  g.degree = level0.d;
  g.neighbors = std::move(level0.data);
  detail::check_ids(graph0_path, g.neighbors, g.n, true);

  std::ifstream in(upper_path, std::ios::binary | std::ios::ate);
  if (!in)
    detail::fail(upper_path, "cannot open");
  uint64_t remaining = uint64_t(in.tellg());
  in.seekg(0);
  uint32_t header[2]; // entry point, max level
  detail::read_values(in, upper_path, header, 2);
  remaining -= 8;
  g.entry_point = header[0];
  if (g.entry_point >= g.n)
    detail::fail(upper_path, "entry point out of range");
  if (header[1] > 64)
    detail::fail(upper_path, "implausible max level");

  for (uint32_t level = 1; level <= header[1]; ++level) {
    const std::string where = " at level " + std::to_string(level);
    uint32_t shape[2]; // node count, degree
    detail::read_values(in, upper_path, shape, 2);
    remaining -= 8;
    // Checked against the bytes left before anything is allocated.
    if (shape[0] == 0 || shape[0] > g.n || shape[1] == 0 || shape[1] > 4096 ||
        shape[0] > remaining / sizeof(VectorId) / (1 + uint64_t(shape[1])))
      detail::fail(upper_path, "bad shape" + where);
    UpperLevel up;
    up.degree = shape[1];
    up.nodes.resize(shape[0]);
    up.neighbors.resize(size_t(shape[0]) * shape[1]);
    detail::read_values(in, upper_path, up.nodes.data(), up.nodes.size());
    detail::read_values(in, upper_path, up.neighbors.data(),
                        up.neighbors.size());
    remaining -= (up.nodes.size() + up.neighbors.size()) * sizeof(VectorId);

    detail::check_ids(upper_path, up.nodes, g.n, false);
    for (size_t i = 1; i < up.nodes.size(); ++i)
      if (up.nodes[i - 1] >= up.nodes[i])
        detail::fail(upper_path, "nodes not sorted and unique" + where);
    if (!g.upper.empty())
      for (VectorId node : up.nodes)
        if (!detail::contains(g.upper.back().nodes, node))
          detail::fail(upper_path, "node missing from level below" + where);
    for (VectorId v : up.neighbors)
      if (v != INVALID_ID && !detail::contains(up.nodes, v))
        detail::fail(upper_path, "neighbor not on its level" + where);
    g.upper.push_back(std::move(up));
  }
  if (remaining != 0)
    detail::fail(upper_path, "trailing bytes");
  if (!g.upper.empty() &&
      !detail::contains(g.upper.back().nodes, g.entry_point))
    detail::fail(upper_path, "entry point not on the top level");
  return g;
}

} // namespace hvs
