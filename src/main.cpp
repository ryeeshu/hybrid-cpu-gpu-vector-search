/**
 * Hybrid CPU-GPU graph-based vector search
 * Christian Yu (cjyu), R Yeeshu Dhurandhar (rdhurand)
 */

#include "options.h"
#include "runner.h"

#include <cstdio>
#include <exception>

int main(int argc, char **argv) {
  try {
    const Options o = parse(argc, argv);
    if (o.print_build_id) {
      std::printf("%s\n", build_id().c_str());
      return 0;
    }
    check_build(o);
    const Dataset data = load_dataset(o);
    if (!o.visit_counts.empty()) {
      write_visit_counts(o, data);
      return 0;
    }
    ReportRow row = launch_row(o, data);
    const auto backends = build_backends(o, data, row);
    run_benchmark(o, data, row, *backends);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    usage();
    return 1;
  }
  return 0;
}
