add_executable(test_types test_types.cpp)
target_link_libraries(test_types PRIVATE hvs)
add_test(NAME types COMMAND test_types)

# Synthetic datasets for tests, generated once per build tree. The generator
# needs numpy, which is checked here so a missing module fails at configure
# time with a clear message.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy"
                RESULT_VARIABLE HVS_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
if(HVS_NUMPY_MISSING)
  message(FATAL_ERROR "${Python3_EXECUTABLE} has no numpy; the tests need it. "
                      "Set -DPython3_EXECUTABLE to a Python that does.")
endif()

set(HVS_GENERATOR ${PROJECT_SOURCE_DIR}/tools/make_synthetic.py)
set(HVS_SYNTHETIC_FILES base.fbin query.fbin gt.ibin graph0.bin upper.bin)
foreach(ef 10 32 64)
  list(APPEND HVS_SYNTHETIC_FILES ref_ef${ef}.ibin ref_ef${ef}_work.ibin)
endforeach()

# hvs_synthetic(<name> <generator args...>) generates build/<name>/. Every
# set has 128 queries, so an 8-thread search, which hands out queries in
# chunks of 16, has a chunk for every thread.
function(hvs_synthetic name)
  set(dir ${PROJECT_BINARY_DIR}/${name})
  list(TRANSFORM HVS_SYNTHETIC_FILES PREPEND ${dir}/ OUTPUT_VARIABLE outputs)
  add_custom_command(
    OUTPUT ${outputs}
    COMMAND ${Python3_EXECUTABLE} ${HVS_GENERATOR} --out ${dir} --nq 128
            ${ARGN}
    DEPENDS ${HVS_GENERATOR}
    COMMENT "Generating ${name} test dataset")
  add_custom_target(${name}_data ALL DEPENDS ${outputs})
endfunction()

# Default set, a low-dimensional one where exact distance ties are common, and
# a small one where searches run out of graph, so many expansions find nothing
# new.
hvs_synthetic(synthetic)
hvs_synthetic(synthetic_ties --d 6 --sigma 4 --clusters 8)
hvs_synthetic(synthetic_small --n 200)
set(HVS_DATA synthetic_data synthetic_ties_data synthetic_small_data)
set(HVS_SMALL ${PROJECT_BINARY_DIR}/synthetic_small)

add_executable(test_io test_io.cpp)
target_link_libraries(test_io PRIVATE hvs)
add_dependencies(test_io ${HVS_DATA})
# Separate scratch directories, so the runs are safe under ctest -j.
add_test(NAME io COMMAND test_io ${PROJECT_BINARY_DIR}/scratch_io
                         ${PROJECT_BINARY_DIR}/synthetic)
add_test(NAME io_ties COMMAND test_io ${PROJECT_BINARY_DIR}/scratch_io_ties
                              ${PROJECT_BINARY_DIR}/synthetic_ties)
add_test(NAME io_small COMMAND test_io ${PROJECT_BINARY_DIR}/scratch_io_small
                               ${HVS_SMALL})

add_executable(test_search test_search.cpp)
target_link_libraries(test_search PRIVATE hvs)
add_dependencies(test_search ${HVS_DATA})
add_test(NAME search COMMAND test_search ${PROJECT_BINARY_DIR}/synthetic)
add_test(NAME search_ties COMMAND test_search
                                  ${PROJECT_BINARY_DIR}/synthetic_ties)
add_test(NAME search_small COMMAND test_search ${HVS_SMALL})

# test_hybrid has a CUDA section, compiled when hvs_gpu is linked. --exhausts
# marks the set whose searches must include expansions that find nothing new.
add_executable(test_hybrid test_hybrid.cpp)
target_link_libraries(test_hybrid PRIVATE hvs)
add_dependencies(test_hybrid ${HVS_DATA})
add_test(NAME hybrid COMMAND test_hybrid ${PROJECT_BINARY_DIR}/synthetic)
add_test(NAME hybrid_ties COMMAND test_hybrid
                                  ${PROJECT_BINARY_DIR}/synthetic_ties)
add_test(NAME hybrid_small COMMAND test_hybrid ${HVS_SMALL} --exhausts)
# These runs need all 8 threads they ask for, whatever OpenMP settings the
# shell has.
set_tests_properties(hybrid hybrid_ties hybrid_small PROPERTIES
  ENVIRONMENT_MODIFICATION
  "OMP_DYNAMIC=set:false;OMP_THREAD_LIMIT=unset:;OMP_NUM_THREADS=unset:")

if(HVS_CUDA)
  target_link_libraries(test_hybrid PRIVATE hvs_gpu)

  add_executable(test_cuda_backend test_cuda_backend.cpp)
  target_link_libraries(test_cuda_backend PRIVATE hvs hvs_gpu)
  add_dependencies(test_cuda_backend ${HVS_DATA})
  add_test(NAME cuda_backend COMMAND test_cuda_backend
                                     ${PROJECT_BINARY_DIR}/synthetic)
  add_test(NAME cuda_backend_ties COMMAND test_cuda_backend
                                          ${PROJECT_BINARY_DIR}/synthetic_ties)
  add_test(NAME cuda_backend_small COMMAND test_cuda_backend ${HVS_SMALL})
  # One GPU test at a time, also under ctest -j: the backend test checks
  # stage times, which another process using the GPU would stretch.
  set_tests_properties(cuda_backend cuda_backend_ties cuda_backend_small
                       hybrid hybrid_ties hybrid_small
                       PROPERTIES RESOURCE_LOCK gpu)
endif()

# Options, CSV report, hashes and --verify checks of hvs_search.
add_executable(test_cli test_cli.cpp)
target_link_libraries(test_cli PRIVATE hvs_cli)
add_dependencies(test_cli ${HVS_DATA})
add_test(NAME cli COMMAND test_cli ${PROJECT_BINARY_DIR}/scratch_cli
                          ${PROJECT_BINARY_DIR}/synthetic)
# Its launches need the 2 threads they ask for.
set_tests_properties(cli PROPERTIES ENVIRONMENT_MODIFICATION
  "OMP_DYNAMIC=set:false;OMP_THREAD_LIMIT=unset:;OMP_NUM_THREADS=unset:")

# hvs_cli_runs(<prefix> <hvs_search options...>): hvs_search end to end on
# two sets at 1 and 8 threads, and on the first 50 queries of one set. Each
# run writes a fresh CSV in its own scratch directory, and check_csv.py checks
# every row against the reference files.
function(hvs_cli_runs prefix)
  foreach(dataset synthetic synthetic_small synthetic_q50)
    foreach(threads 1 8)
      set(name ${prefix}_${dataset}_t${threads})
      set(data ${dataset})
      set(count 0)
      if(dataset STREQUAL synthetic_q50)
        if(threads EQUAL 1)
          continue()
        endif()
        set(data synthetic)
        set(count 50)
      endif()
      add_test(NAME ${name}
               COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_csv.py
                       --binary $<TARGET_FILE:hvs_search>
                       --data ${PROJECT_BINARY_DIR}/${data}
                       --scratch ${PROJECT_BINARY_DIR}/scratch_${name}
                       --threads ${threads} --query-count ${count} ${ARGN})
      # threads_used must equal threads, whatever OpenMP settings the shell
      # has.
      set_tests_properties(${name} PROPERTIES
        ENVIRONMENT_MODIFICATION
        "OMP_DYNAMIC=set:false;OMP_THREAD_LIMIT=unset:;OMP_NUM_THREADS=unset:")
      # Runs on the GPU take the same lock as the other GPU tests.
      if(prefix MATCHES "cuda")
        set_tests_properties(${name} PROPERTIES RESOURCE_LOCK gpu)
      endif()
    endforeach()
  endforeach()
endfunction()

hvs_cli_runs(cli --mode cpu)
hvs_cli_runs(cli_hybrid_cpu --mode hybrid-sync --backend cpu --verify)
if(HVS_CUDA)
  hvs_cli_runs(cli_hybrid_cuda --mode hybrid-sync --backend cuda --verify)
  hvs_cli_runs(cli_hybrid_cuda_staged --mode hybrid-sync --backend cuda
               --stage-timing --verify)
endif()

# Command lines hvs_search must refuse before loading any data, writing no
# CSV. --backend cuda is one of them only in a build without CUDA.
set(HVS_REJECTS --rejects)
if(NOT HVS_CUDA)
  list(APPEND HVS_REJECTS --no-cuda)
endif()
add_test(NAME cli_rejects
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_csv.py
                 --binary $<TARGET_FILE:hvs_search>
                 --data ${PROJECT_BINARY_DIR}/synthetic
                 --scratch ${PROJECT_BINARY_DIR}/scratch_cli_rejects
                 --threads 1 ${HVS_REJECTS})

# plot.py, check_results.py and visit_cdf.py on small hand-written fixtures.
# The plot.py part needs pandas and matplotlib; where the Python above has
# neither, the test prints SKIPPED for that part and runs the rest.
add_test(NAME tools
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_tools.py
                 ${PROJECT_BINARY_DIR}/scratch_tools)
