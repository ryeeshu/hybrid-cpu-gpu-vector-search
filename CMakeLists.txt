cmake_minimum_required(VERSION 3.22)
project(hvs LANGUAGES CXX)

option(HVS_CUDA "Build the CUDA distance backend" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header-only core. AVX2 + FMA is the common baseline of both machines, so
# -march=native is deliberately not used.
add_library(hvs INTERFACE)
target_include_directories(hvs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(HVS_CXX_OPTIONS -Wall -Wextra -mavx2 -mfma)
target_compile_options(hvs INTERFACE ${HVS_CXX_OPTIONS})

find_package(OpenMP REQUIRED)
target_link_libraries(hvs INTERFACE OpenMP::OpenMP_CXX)

# Options, CSV report and benchmark runner of hvs_search, in a library so the
# tests can link them.
add_library(hvs_cli STATIC src/options.cpp src/report.cpp src/runner.cpp)
target_include_directories(hvs_cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(hvs_cli PUBLIC hvs)

add_executable(hvs_search src/main.cpp)
target_link_libraries(hvs_search PRIVATE hvs)
target_link_libraries(hvs_search PRIVATE hvs_cli)

if(HVS_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_library(hvs_gpu STATIC src/gpu/cuda_backend.cu)
  target_include_directories(hvs_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # Lets C++ sources that link hvs_gpu test for it with #ifdef HVS_CUDA.
  target_compile_definitions(hvs_gpu PUBLIC HVS_CUDA)
  # hvs_search runs hybrid-sync with CUDA backends through the library.
  target_link_libraries(hvs_cli PUBLIC hvs_gpu)

  add_executable(xfer_bench benchmarks/xfer_bench.cpp)
  # hvs_cli gives it the dataset loader, provenance columns and CSV checks of
  # hvs_search.
  target_link_libraries(xfer_bench PRIVATE hvs hvs_gpu hvs_cli)
endif()

# The build id is recomputed from the sources and the build settings on
# every build, so a binary never reports the sources of an older build. The
# flags are the effective ones: CMAKE_CXX_FLAGS (which holds CXXFLAGS from the
# environment of the first configure), the build type's flags and our own.
string(TOUPPER "${CMAKE_BUILD_TYPE}" HVS_CONFIG)
list(JOIN HVS_CXX_OPTIONS " " HVS_FLAGS)
set(HVS_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${HVS_CONFIG}} ${HVS_FLAGS}")
set(HVS_CUDA_COMPILER "")
set(HVS_CUDA_FLAGS "")
if(HVS_CUDA)
  set(HVS_CUDA_COMPILER
      "${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}")
  # The architectures are not a flag here, but change the binary like one.
  set(HVS_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CMAKE_CUDA_FLAGS_${HVS_CONFIG}} \
CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()
# Single spaces, none at either end.
foreach(flags HVS_FLAGS HVS_CUDA_FLAGS)
  string(REGEX REPLACE " +" " " ${flags} "${${flags}}")
  string(STRIP "${${flags}}" ${flags})
endforeach()
set(HVS_GENERATED ${PROJECT_BINARY_DIR}/generated)
add_custom_target(hvs_build_id
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR}
          -DOUTPUT=${HVS_GENERATED}/build_id.h
          -DMANIFEST=${PROJECT_BINARY_DIR}/build_manifest.txt
          -DBUILD_TYPE=${CMAKE_BUILD_TYPE}
          "-DCXX_COMPILER=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
          -DCXX_FLAGS=${HVS_FLAGS} -DCUDA=${HVS_CUDA}
          -DCUDA_COMPILER=${HVS_CUDA_COMPILER} -DCUDA_FLAGS=${HVS_CUDA_FLAGS}
          -P ${PROJECT_SOURCE_DIR}/tools/build_id.cmake
  BYPRODUCTS ${HVS_GENERATED}/build_id.h
             ${PROJECT_BINARY_DIR}/build_manifest.txt
  COMMENT "Computing the build id"
  VERBATIM)
add_dependencies(hvs_cli hvs_build_id)
target_include_directories(hvs_cli PRIVATE ${HVS_GENERATED})

enable_testing()
add_subdirectory(tests)
