/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "options.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

void usage() {
  std::fprintf(
      stderr,
      "usage: hvs_search --data DIR [--mode cpu|hybrid-sync] [--k N] "
      "[--ef A,B,..]\n"
      "                  [--threads N] [--query-count N] "
      "[--repetitions N]\n"
      "                  [--csv FILE] [--tag TEXT] [--launch N] "
      "[--visit-counts FILE]\n"
      "                  [--visit-range FIRST,COUNT] [--backend cuda|cpu] "
      "[--stage-timing]\n"
      "                  [--verify]\n"
      "       hvs_search --build-id\n"
      "  --build-id      print the id of the sources and settings of this "
      "build\n"
      "  --backend       distance backend of hybrid-sync (default cuda)\n"
      "  --stage-timing  hybrid-sync with cuda: synchronize after each GPU "
      "stage to time it\n"
      "  --verify        hybrid-sync: also run V1 and fail unless the "
      "results are identical\n"
      "  --tag           label written to every CSV row; no commas, "
      "quotes or newlines\n"
      "  --launch        index written to every CSV row, to tell repeated "
      "launches apart\n"
      "  --visit-counts  untimed run over DIR/learn.fbin that writes "
      "how often each\n"
      "                  vector's distance was computed (uses the "
      "first --ef value)\n"
      "  --visit-range   with --visit-counts: use COUNT learn rows from "
      "FIRST (default all)\n");
}

uint32_t parse_u32(const std::string &flag, const std::string &value) {
  if (value.empty() || value.find_first_not_of("0123456789") != value.npos)
    throw std::invalid_argument(flag + " needs a number, not '" + value + "'");
  uint64_t number = 0;
  for (char c : value) {
    number = number * 10 + uint64_t(c - '0');
    if (number > UINT32_MAX)
      throw std::invalid_argument(flag + " value " + value + " is too large");
  }
  return uint32_t(number);
}

// Reads a comma-separated list of distinct positive ef values.
static std::vector<uint32_t> parse_ef_list(const std::string &value) {
  std::vector<uint32_t> list;
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = std::min(value.find(',', start), value.size());
    const uint32_t ef = parse_u32("--ef", value.substr(start, comma - start));
    if (ef == 0)
      throw std::invalid_argument("--ef values must be positive");
    if (std::count(list.begin(), list.end(), ef) != 0)
      throw std::invalid_argument("--ef lists " + std::to_string(ef) +
                                  " twice");
    list.push_back(ef);
    start = comma + 1;
  }
  return list;
}

Options parse(int argc, char **argv) {
  Options o;
  for (int i = 0; i < argc; ++i)
    o.command += (i ? " " : "") + std::string(argv[i]);
  bool visit_range = false;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    // Flags without a value.
    if (flag == "--build-id") {
      o.print_build_id = true;
      continue;
    }
    if (flag == "--stage-timing") {
      o.stage_timing = true;
      continue;
    }
    if (flag == "--verify") {
      o.verify = true;
      continue;
    }
    if (i + 1 >= argc)
      throw std::invalid_argument("missing value for " + flag);
    const std::string value = argv[++i];
    if (flag == "--data")
      o.data = value;
    else if (flag == "--mode")
      o.mode = value;
    else if (flag == "--backend")
      o.backend = value;
    else if (flag == "--csv")
      o.csv = value;
    else if (flag == "--tag")
      o.tag = value;
    else if (flag == "--visit-counts")
      o.visit_counts = value;
    else if (flag == "--visit-range") {
      const size_t comma = value.find(',');
      if (comma == value.npos)
        throw std::invalid_argument("--visit-range needs FIRST,COUNT");
      o.visit_first = parse_u32(flag, value.substr(0, comma));
      o.visit_count = parse_u32(flag, value.substr(comma + 1));
      if (o.visit_count == 0)
        throw std::invalid_argument("--visit-range COUNT must be positive");
      visit_range = true;
    } else if (flag == "--k")
      o.k = parse_u32(flag, value);
    else if (flag == "--threads")
      o.threads = parse_u32(flag, value);
    else if (flag == "--query-count")
      o.query_count = parse_u32(flag, value);
    else if (flag == "--repetitions")
      o.repetitions = parse_u32(flag, value);
    else if (flag == "--launch")
      o.launch = parse_u32(flag, value);
    else if (flag == "--ef")
      o.ef = parse_ef_list(value);
    else
      throw std::invalid_argument("unknown flag " + flag);
  }
  if (o.print_build_id)
    return o; // needs nothing else
  if (o.data.empty())
    throw std::invalid_argument("--data is required");
  if (visit_range && o.visit_counts.empty())
    throw std::invalid_argument("--visit-range needs --visit-counts");
  if (o.k == 0 || o.threads == 0 || o.repetitions == 0)
    throw std::invalid_argument(
        "--k, --threads and --repetitions must be positive");
  // The tag is written into the CSV unquoted.
  if (o.tag.empty() || o.tag.find_first_of(",\"'\r\n") != o.tag.npos)
    throw std::invalid_argument(
        "--tag must be non-empty, without commas, quotes or newlines");
  if (o.mode != "cpu" && o.mode != "hybrid-sync")
    throw std::invalid_argument("--mode must be cpu or hybrid-sync");
  if (o.mode == "cpu" && !o.backend.empty())
    throw std::invalid_argument("--backend needs --mode hybrid-sync");
  if (o.mode == "hybrid-sync" && o.backend.empty())
    o.backend = "cuda";
  if (o.mode == "hybrid-sync" && o.backend != "cuda" && o.backend != "cpu")
    throw std::invalid_argument("--backend must be cuda or cpu");
  if (o.stage_timing && o.backend != "cuda")
    throw std::invalid_argument(
        "--stage-timing needs --mode hybrid-sync with --backend cuda");
  if (o.verify && o.mode == "cpu")
    throw std::invalid_argument(
        "--verify needs --mode hybrid-sync; cpu mode has nothing to compare");
  return o;
}
