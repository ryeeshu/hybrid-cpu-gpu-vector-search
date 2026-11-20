/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "check.h"
#include "hvs/io.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using namespace hvs;

static void write_file(const std::string &path,
                       const std::vector<uint32_t> &words) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(words.data()),
            std::streamsize(words.size() * sizeof(uint32_t)));
}

// True if loading throws, which is what a malformed file must do.
template <typename Load> static bool rejects(Load &&load) {
  try {
    load();
  } catch (const std::runtime_error &) {
    return true;
  }
  return false;
}

static void test_round_trip_and_rejection(const std::string &tmp) {
  const std::string g0 = tmp + "/g0.bin", up = tmp + "/up.bin";
  const VectorId X = INVALID_ID;
  // 4 nodes, degree 2, with a hole in the middle of row 1.
  write_file(g0, {4, 2, 1, 2, X, 0, 0, 1, 2, 0});
  // Entry point 2. Level 1: nodes {0, 2, 3}, degree 2. Level 2: node {2},
  // degree 1, with an empty list.
  const std::vector<uint32_t> good = {2, 2, 3, 2, 0, 2, 3, 2, X,
                                      0, 3, 2, X, 1, 1, 2, X};
  write_file(up, good);
  const Graph g = load_graph(g0, up);
  CHECK(g.n == 4 && g.degree == 2 && g.entry_point == 2);
  CHECK(g.row(1)[0] == X && g.row(1)[1] == 0);
  CHECK(g.upper.size() == 2);
  CHECK(g.upper[0].degree == 2 && g.upper[0].nodes.size() == 3);
  CHECK(g.upper[0].nodes[1] == 2 && g.upper[0].neighbors[2] == 0);
  CHECK(g.upper[1].degree == 1 && g.upper[1].nodes[0] == 2);
  CHECK(g.upper[1].neighbors[0] == X);

  const std::string bad = tmp + "/bad.bin";
  CHECK(rejects([&] { load_matrix<float>(tmp + "/missing.bin"); }));
  write_file(bad, {3, 2, 1, 2, 3}); // shorter than the header claims
  CHECK(rejects([&] { load_matrix<VectorId>(bad); }));
  write_file(bad, {1, 2, 0, 0, 0}); // longer than the header claims
  CHECK(rejects([&] { load_matrix<VectorId>(bad); }));
  write_file(bad, {0, 2}); // no rows
  CHECK(rejects([&] { load_matrix<VectorId>(bad); }));
  write_file(bad, {2, 0}); // no columns
  CHECK(rejects([&] { load_matrix<VectorId>(bad); }));
  write_file(bad, {0x80000000u, 0x80000000u}); // n * d * 4 wraps in 64 bits
  CHECK(rejects([&] { load_matrix<VectorId>(bad); }));
  write_file(bad, {4, 2, 1, 2, 4, 0, 0, 1, 2, 0}); // neighbor id 4 with n = 4
  CHECK(rejects([&] { load_graph(bad, up); }));

  // Each case is the good upper file with one thing wrong.
  auto upper_rejected = [&](const std::vector<uint32_t> &words) {
    write_file(bad, words);
    return rejects([&] { load_graph(g0, bad); });
  };
  auto with = [&](size_t index, uint32_t value) {
    std::vector<uint32_t> words = good;
    words[index] = value;
    return words;
  };
  CHECK(upper_rejected({2}));              // truncated header
  CHECK(upper_rejected(with(0, 4)));       // entry point out of range
  CHECK(upper_rejected(with(0, 3)));       // entry point not on the top level
  CHECK(upper_rejected(with(1, 65)));      // implausible max level
  CHECK(upper_rejected(with(1, 3)));       // claims a level that is absent
  CHECK(upper_rejected(with(2, 0)));       // level with no nodes
  CHECK(upper_rejected(with(2, 1000000))); // node count beyond the file
  CHECK(upper_rejected(with(3, 0)));       // level with degree 0
  CHECK(upper_rejected(with(4, 4)));       // node id out of range
  CHECK(upper_rejected(with(4, X)));       // INVALID as a node id
  CHECK(upper_rejected(with(5, 0)));       // nodes not strictly increasing
  CHECK(upper_rejected(with(7, 1)));       // neighbor not on its level
  CHECK(upper_rejected(with(7, 4)));       // neighbor id out of range
  CHECK(upper_rejected(with(15, 1)));      // level 2 node missing below
  // Cases built so that only the named check can reject them.
  auto with2 = [&](size_t i, uint32_t a, size_t j, uint32_t b) {
    std::vector<uint32_t> words = with(i, a);
    words[j] = b;
    return words;
  };
  CHECK(upper_rejected({4, 0}));             // entry range, no upper levels
  CHECK(upper_rejected(with2(6, 2, 10, 0))); // duplicate node {0, 2, 2}
  CHECK(upper_rejected(with2(6, 7, 10, 0))); // node out of range {0, 2, 7}
  CHECK(upper_rejected(with2(6, X, 10, 0))); // INVALID node {0, 2, X}
  CHECK(upper_rejected(with2(15, 1, 0, 1))); // level 2 not a subset
  CHECK(upper_rejected({2, 1, 1, 0, 2}));    // degree 0, nothing else wrong
  std::vector<uint32_t> changed = good;
  changed.push_back(7);
  CHECK(upper_rejected(changed)); // trailing bytes
  changed = good;
  changed.pop_back();
  CHECK(upper_rejected(changed)); // truncated list
}

// Brute force over the synthetic set must reproduce its ground truth exactly,
// including the (distance, id) order: coordinates are integers.
static void test_ground_truth(const std::string &data) {
  const FloatMatrix base = load_matrix<float>(data + "/base.fbin");
  const FloatMatrix queries = load_matrix<float>(data + "/query.fbin");
  const IdMatrix gt = load_matrix<VectorId>(data + "/gt.ibin");
  const Graph g = load_graph(data + "/graph0.bin", data + "/upper.bin");
  CHECK(base.d == queries.d && gt.n == queries.n && g.n == base.n);

  std::vector<std::pair<float, VectorId>> all(base.n);
  for (uint32_t q = 0; q < queries.n; ++q) {
    for (uint32_t i = 0; i < base.n; ++i) {
      float sum = 0.0f;
      for (uint32_t j = 0; j < base.d; ++j) {
        const float diff = queries.row(q)[j] - base.row(i)[j];
        sum += diff * diff;
      }
      all[i] = {sum, i};
    }
    std::partial_sort(all.begin(), all.begin() + gt.d, all.end());
    for (uint32_t r = 0; r < gt.d; ++r)
      CHECK(all[r].second == gt.row(q)[r]);
  }
}

int main(int argc, char **argv) {
  CHECK(argc == 3); // scratch directory, synthetic data directory
  std::filesystem::create_directories(argv[1]);
  test_round_trip_and_rejection(argv[1]);
  test_ground_truth(argv[2]);
  std::printf("test_io passed\n");
  return 0;
}
