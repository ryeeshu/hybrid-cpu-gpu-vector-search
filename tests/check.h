/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#pragma once

#include <cstdio>
#include <cstdlib>

// Always-on test assertion. assert() compiles out under NDEBUG, which would
// let Release-build tests pass without checking anything.
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__,    \
                   #cond);                                                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
