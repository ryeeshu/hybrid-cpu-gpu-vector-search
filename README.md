# hybrid-cpu-gpu-vector-search

Heterogeneous CPU-GPU graph-based vector search with memory-aware execution.
CMU 15-618 final project, Fall 2026: R Yeeshu Dhurandhar (rdhurand) and
Christian Yu (cjyu).

Project page: https://ryeeshu.github.io/hybrid-vector-search/

## Build

Requires CMake 3.22 or newer, a C++17 compiler with AVX2 and FMA, and
Python 3 with numpy (the tests generate a small dataset at build time).
CUDA is optional.

    cmake --preset cpu && cmake --build --preset cpu && ctest --preset cpu

Presets:

| Preset | Use |
|---|---|
| `cpu` | CPU only, no CUDA required |
| `wsl` | Desktop under WSL2 with a CUDA 12.x toolkit, build tree in `~/hvs-build` |
| `ghc` | GHC machines: CUDA 11.7 with `g++-11`, build tree in `/tmp` |

## Data

The search engine reads its own binary files and does not link against Faiss.
Faiss is needed only to build and export an index:

    python3 -m venv ~/hvs-venv && ~/hvs-venv/bin/pip install numpy faiss-cpu
    tools/get_sift1m.sh ~/hvs-data
    ~/hvs-venv/bin/python tools/export_hnsw.py \
        --base ~/hvs-data/sift/sift_base.fvecs \
        --query ~/hvs-data/sift/sift_query.fvecs \
        --gt ~/hvs-data/sift/sift_groundtruth.ivecs \
        --learn ~/hvs-data/sift/sift_learn.fvecs \
        --out ~/hvs-data/sift1m_m16

File formats, all little-endian:

| File | Contents |
|---|---|
| `base.fbin`, `query.fbin`, `learn.fbin` | `uint32 n, uint32 d`, then `n * d` float32 |
| `gt.ibin` | `uint32 n, uint32 k`, then `n * k` uint32 ids |
| `graph0.bin` | `uint32 n, uint32 degree`, then `n * degree` uint32 neighbor ids of HNSW level 0; `0xFFFFFFFF` marks an empty slot |
| `upper.bin` | `uint32 entry_point, uint32 max_level`, then for each level 1..max_level: `uint32 count, uint32 degree`, `count` sorted node ids, `count * degree` neighbor ids |
| `faiss_reference.json` | Recall and distance evaluations per query of Faiss's own search, per efSearch |

## Benchmark CSV

    hvs_search --data ~/hvs-data/sift1m_m16 --ef 16,64 --threads 8 \
        --repetitions 3 --tag sweep1 --launch 1 --csv results.csv

    hvs_search --data ~/hvs-data/sift1m_m16 --mode hybrid-sync --verify \
        --ef 64 --threads 8 --repetitions 1 --csv results.csv

`--mode cpu` (the default) runs V1, the CPU-only search. `--mode
hybrid-sync` runs V2, the synchronous hybrid: each CPU thread walks the
graph as in V1, but every batch of candidate distances is computed by one
blocking call to that thread's own distance backend. `--backend cuda` (the
default) computes them on the GPU; `--backend cpu` computes them on the
calling thread, so a build without CUDA can run the mode. `--backend cuda`
in a build without CUDA fails before any data is loaded. The backends are
built once per launch, one per thread, and given the queries before any
pass. A CUDA backend ships the candidate vectors from pinned host memory
(not resident) with at most one round trip per batch (`max_batch` = graph
degree).

`--stage-timing` (with `--backend cuda` only) adds a stream synchronization
after each GPU stage (copy in, kernel, copy back) so each can be timed.
Depending on the platform and the thread count, a stage-timing run can be
slower or faster than a throughput run, so its QPS is never a throughput
figure. `--verify` (hybrid-sync only) also runs V1 in the
same process, once per ef with the same queries, k, ef and thread count, and
fails the launch unless all of these hold:

- every timed repetition's `result_hash` equals V1's;
- in the last repetition, every query has V1's ids, distances bit for bit,
  distance evaluations, hops and batches;
- the number of queries compared is positive and equals the number searched;
- every result list on both sides has min(k, ef) entries (fewer only if the
  graph has fewer vertices);
- in that repetition, the candidates summed over backends equal the summed
  distance evaluations, and the round trips (calls, for the cpu backend)
  equal the summed batches;
- `max_batch` is at least the graph degree, so no batch is split.

The bitwise distance comparison holds on integer-valued data sets such as
SIFT1M and the generated test sets, where CPU and GPU sums are exact; on
other data their distances can differ in the last bits. A failed check is
reported with the query it failed at, and no row of the launch is written.
`verified = yes` means that V2 returned exactly what V1 returned in the same
process, not that either is right; that is what the tests against the
reference files and the expected result hashes are for. A wrong distance
that changes no result, evaluation, hop or batch of any repetition, and no
distance of the last one, is not detected.

Each ef runs one discarded warm-up pass, then `--repetitions` timed passes,
and every timed pass is one CSV row. The console shows, per ef, the
repetition with the median QPS. Rows are appended only after every ef has
run; nothing is written if the file's first line is not the current header,
if the search ran on fewer threads than asked for, if repetitions disagree
on `result_hash`, or if `--verify` fails. The file is checked at startup,
before any data is loaded: a different header or a path that cannot be
opened for appending fails at once, and a missing file is not created until
the rows are written. A missing or empty file gets the header first. Every
mode writes the same 61 columns; a column that does not apply to a row is
left empty.

| Column | Meaning |
|---|---|
| `build_id` | Id of the sources and build settings of the binary (below) |
| `tag` | `--tag` label (default `untagged`) |
| `host` | Host name |
| `platform` | `wsl2` if `/proc/version` mentions Microsoft, else `linux` |
| `build` | `release` or `debug` (whether `NDEBUG` was set) |
| `compiler` | Compiler and version, such as `gcc-13.3.0` |
| `build_type` | `CMAKE_BUILD_TYPE`, such as `Release` or `RelWithDebInfo` |
| `cxx_flags` | Effective C++ flags: `CMAKE_CXX_FLAGS` (which holds `CXXFLAGS` from the environment of the first configure), those of the build type, then the project's own; commas become `;`, quotes `_` |
| `logical_cpus` | `std::thread::hardware_concurrency()` |
| `omp_proc_bind`, `omp_places` | Those environment variables, or `unset`; commas become `;` |
| `dataset_hash` | 64-bit FNV-1a over the bytes of `base.fbin`, `query.fbin`, `graph0.bin`, `upper.bin` and `gt.ibin`, in that order |
| `n`, `d`, `degree` | Base vectors, dimension, level-0 graph degree |
| `queries` | Queries searched (all of `query.fbin`, or `--query-count`) |
| `k`, `ef` | Results per query and search list size |
| `mode` | `cpu`: V1, CPU-only search; `hybrid-sync`: V2, the synchronous hybrid |
| `threads` | Threads asked for with `--threads` |
| `threads_used` | Threads the search ran on; always equal to `threads` |
| `launch` | `--launch` index (default 1), to tell repeated launches apart |
| `repetition` | Timed pass within this launch and ef, from 1 |
| `warmups` | Discarded passes before the timed ones (1) |
| `recall` | Mean fraction of the true k nearest (`gt.ibin`) found |
| `qps` | Queries divided by the wall time of the pass |
| `latency_mean_us` | Mean wall time of one query in microseconds, measured inside the closed-loop pass, so it is latency at a concurrency of `threads`, not service time in isolation |
| `latency_p50_us`, `latency_p95_us`, `latency_p99_us` | Nearest-rank percentiles of the same sample: element ceil(p * count / 100) of the sorted latencies |
| `evals_per_query` | Distance evaluations per query |
| `hops_per_query` | Expanded candidates per query |
| `batches_per_query` | Distance batches per query (entry point plus each expansion that found a new vector) |
| `result_hash` | 64-bit FNV-1a, 16 lowercase hex digits, over little-endian uint32 values: the query count, then per query in id order its id, result count, result ids, distance evaluations and hops. Distances are not included, so it is the same on every machine, compiler and thread count |
| `backend` | `cuda` or `cpu` on hybrid-sync rows, else empty |
| `stage_timing` | 1 with `--stage-timing`, else 0; empty in cpu mode |
| `verified` | `yes` only if `--verify` ran and passed for this launch, else `no` |
| `gpu` | Device name |
| `cuda_runtime`, `cuda_driver` | CUDA runtime and driver versions, such as `12.6` |
| `wait_policy` | `auto`: the device scheduling flags are left at the CUDA default |
| `max_batch` | Candidates per round trip; the graph degree, so a batch is never split |
| `pinned`, `resident` | Whether the host staging buffers are page-locked (always 1) and whether the base vectors stay on the device (always 0) |
| `init_s` | Wall seconds to build the backends and give them the queries, once per launch |
| `device_bytes` | Device memory allocated, summed over backends |
| `launch_floor_us` | Mean time to launch an empty kernel and wait for it, measured once per launch outside the timed passes |
| `round_trips_per_query` | Backend round trips per query (calls, for the cpu backend); equals `batches_per_query` |
| `candidates_per_round_trip` | Candidates per round trip |
| `bytes_to_device_per_query` | Bytes copied to the device per query: 4 + 4 d per candidate |
| `bytes_from_device_per_query` | Bytes copied back per query: 4 per candidate |
| `backend_us` | Time inside the backend's round trips, summed over backends, per query |
| `outside_backend_us` | Sum of the query latencies minus the time inside round trips, per query: a residual that contains graph traversal plus the glue around each backend call |
| `gather_us` | Host time filling the staging buffers, also with `--stage-timing` |
| `submit_us` | Host time inside the copy and launch calls; not necessarily cheap |
| `wait_us` | Host time blocked in the final stream synchronize, not GPU time |
| `to_device_us`, `kernel_us`, `from_device_us` | Copy in, kernel and copy back, each ending in a stream synchronize |
| `start_time_utc` | When the launch started, after loading the data, such as `2026-11-19T12:34:56Z` |
| `command` | The command line, space-joined; commas become `;`, quotes and line breaks `_` |

The columns from `backend` to `from_device_us` are filled as follows; every
other row leaves them empty. All rows: `verified`. hybrid-sync rows:
`backend`, `stage_timing`, `init_s`, `round_trips_per_query`,
`candidates_per_round_trip`. Rows of the cuda backend, in addition: `gpu`
through `resident`, `device_bytes`, `launch_floor_us`, both bytes columns,
`backend_us`, `outside_backend_us` and `gather_us`, then `submit_us` and
`wait_us` without `--stage-timing`, or `to_device_us`, `kernel_us` and
`from_device_us` with it. The cpu backend measures no times or bytes.

All `*_us` backend columns are per query and are thread time summed over
threads, not wall time: with T threads they add up to about T / QPS.
`backend_us + outside_backend_us` is `latency_mean_us`, and
`gather_us + submit_us + wait_us` is at most `backend_us`, and close to it:
the rest is copying the results out of the staging buffer. Rows with
`stage_timing` = 1 are a separate run type: there `backend_us` is about
`gather_us + to_device_us + kernel_us + from_device_us`, each stage includes
a stream synchronization the throughput run does not perform, so their QPS
is not a throughput figure and their stage times are not comparable with
submit and wait.

A V2 row measures one design point, the synchronous hybrid with one blocking
round trip per expansion, on the same work as V1: with `--verify` the
results, evaluations, hops and batches are shown to be identical. Compare
V2 with V1 only between rows of equal `dataset_hash`, `queries`, `k`, `ef`
and `result_hash`, and use only rows with `stage_timing` = 0 for QPS. A V2 row is
not a measure of GPU throughput: each round trip carries only one
expansion's candidates (`candidates_per_round_trip`), and
`backend_us / round_trips_per_query`, the mean round trip, is to be read
next to `launch_floor_us`. `outside_backend_us` is a residual, not a
traversal time, and the backend columns are thread time, not wall time.

`build_id` is computed on every build by `tools/build_id.cmake`, as the
first 12 hex digits of the SHA-256 of a manifest. The manifest starts with
one `name=value` line per build setting: `build_type`, `cxx_compiler` (CMake
compiler id and version, such as `GNU 13.3.0`), `cxx_flags` (as in the
column), `cuda` (`HVS_CUDA`, `ON` or `OFF`), `cuda_compiler` and
`cuda_flags` (`CMAKE_CUDA_FLAGS`, those of the build type, and the
architectures; both empty without CUDA). Then comes one line per file under
`include/`, `src/`, `benchmarks/`, `tests/` and `tools/`, plus
`CMakeLists.txt` and `CMakePresets.json`, sorted by relative path, holding
the path and the SHA-256 of the file with CR characters removed. Caches and
editor backups (`__pycache__`, `.pytest_cache`, `*.pyc`, `*.swp`, `*~`) are
left out, and so is this README; `tools/plot.py` is left out too, because it
only post-processes saved CSVs. The id does not use git or absolute paths,
so a copy of the tree with the same settings gives the same id. Every build
writes the manifest to `build_manifest.txt` in the build directory, only
when it changes, and `hvs_search --build-id` prints the id.

The script also runs on its own, to print the id of a source tree for given
settings, and with `-DMANIFEST=FILE` writes the manifest to FILE. For the
`cpu` preset with g++ 13.3:

    cmake -DSOURCE_DIR=. -DBUILD_TYPE=Release "-DCXX_COMPILER=GNU 13.3.0" \
        "-DCXX_FLAGS=-O3 -DNDEBUG -Wall -Wextra -mavx2 -mfma" -DCUDA=OFF \
        -DMANIFEST=/tmp/manifest.txt -P tools/build_id.cmake

The settings of a build are the first lines of its `build_manifest.txt`;
add `-DCUDA_COMPILER=...` and `-DCUDA_FLAGS=...` for a CUDA build.

## Visit counts

    hvs_search --data ~/hvs-data/sift1m_m16 --ef 64 --threads 8 \
        --visit-counts visits.bin --visit-range 0,50000

counts, without timing, how often each base vector's distance is computed
when the rows of `learn.fbin` given by `--visit-range FIRST,COUNT` (default
all) are searched at the first `--ef` value. `visits.bin` is `uint32 n,
uint32 1`, then the n counts. After a summary, stdout has one `key=value`
line each for `ef`, `k`, `query_file`, `range` (FIRST,COUNT),
`evaluations`, `never_visited` (vectors never evaluated), `entry_point`,
`dataset_hash` and `build_id`.

## Transfer benchmark

    xfer_bench --data ~/hvs-data/sift1m_m16 --csv xfer.csv --tag sweep1 --run 1

(CUDA builds only) prices one distance round trip against batch size: 1 to
64 pairs by doubling, then powers of 4 up to `--max-mb` (default 256) MB of
vectors. Each pair is a random query and a random base vector, fresh every
iteration. Each configuration is warmed up for at least 50 ms and 3
iterations, then timed for `iterations` round trips; every figure is the
median over those iterations, within one run. The whole benchmark runs from
one calling thread on an otherwise idle GPU. `--run` (default 1) tells
repeated runs apart; at small batches runs differ more than iterations do.
The CSV is checked at startup and guarded like the `hvs_search` one; the
rows of a run are appended at its end.

| Column | Meaning |
|---|---|
| `build_id` through `omp_places`, `dataset_hash`, `n`, `d`, `start_time_utc` | As in the benchmark CSV above; `gpu` is the device name |
| `cpu_threads` | Threads of the `cpu_all_us` loop: the OpenMP default team size, checked against the team it ran on |
| `run`, `pairs`, `iterations` | `--run`, batch size, timed round trips per configuration |
| `launch_floor_us` | Median time to launch an empty kernel and wait for it |
| `ship_pageable_us` | Round trip shipping the vectors from pageable host memory. A single-caller figure: at small batches it can beat pinned, but with 8 concurrent callers we measured it about 4 times slower |
| `ship_pinned_us` | Round trip shipping the vectors from pinned host memory |
| `gather_us` | The host gather of that pinned run alone; part of `ship_pinned_us` |
| `staged_round_trip_us` | Round trip of a separate pinned run with a synchronize after each stage |
| `to_device_us`, `kernel_us`, `from_device_us` | The stages of that run; with `gather_us` they add up to about `staged_round_trip_us` |
| `resident_us` | Round trip shipping only ids to a device that holds every base vector |
| `cpu_1t_us`, `cpu_all_us` | The CPU computing the same distances with 1 and `cpu_threads` threads |

## Layout

    include/hvs/   shared types, the distance-backend interface, file loaders
    src/gpu/       CUDA kernels
    tests/         CTest programs
    tools/         dataset download, Faiss export, synthetic test data
