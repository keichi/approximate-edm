cmake_minimum_required(VERSION 3.16)
project(approx-edm CXX)

enable_language(CUDA)

add_executable(approx-edm approx_edm.cpp)
add_executable(eval-hnsw-edm eval_hnsw_edm.cpp)

find_package(OpenMP REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(faiss REQUIRED)
find_package(nanoflann REQUIRED)

set(DEPS OpenMP::OpenMP_CXX xtl xtensor faiss CUDA::cublas nanoflann::nanoflann)

target_link_libraries(approx-edm PRIVATE ${DEPS})
target_link_libraries(eval-hnsw-edm PRIVATE ${DEPS})
