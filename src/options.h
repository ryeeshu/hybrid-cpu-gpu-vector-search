/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Command-line settings of hvs_search.
struct Options {
  std::string data;
  std::string mode = "cpu";
  std::string csv;
  std::string tag = "untagged";
  std::string visit_counts;
  uint32_t visit_first = 0; // --visit-range: first learn row used
  uint32_t visit_count = 0; // and how many; 0 means to the end
  std::string command;      // the command line, space-joined
  bool print_build_id = false;
  std::string backend;       // hybrid-sync only: cuda (the default) or cpu
  bool stage_timing = false; // hybrid-sync with cuda: time each GPU stage
  bool verify = false;       // hybrid-sync only: compare the results with V1
  std::vector<uint32_t> ef = {16, 32, 64, 128, 256};
  uint32_t k = 10;
  uint32_t threads = 1;
  uint32_t query_count = 0; // 0 means every query
  uint32_t repetitions = 3;
  uint32_t launch = 1; // tells repeated launches of one configuration apart
};

// Prints the usage text to stderr.
void usage();

// Reads a decimal number that fits in 32 bits: digits only, no sign, no
// spaces. Throws std::invalid_argument naming flag otherwise.
uint32_t parse_u32(const std::string &flag, const std::string &value);

// Reads the command line. Throws std::invalid_argument if it is malformed.
Options parse(int argc, char **argv);
