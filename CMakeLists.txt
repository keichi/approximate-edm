cmake_minimum_required(VERSION 3.16)
project(approx-edm CXX)

enable_language(CUDA)

find_package(OpenMP REQUIRED)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(faiss REQUIRED)
find_package(nanoflann REQUIRED)

set(DEPS OpenMP::OpenMP_CXX xtl xtensor faiss CUDA::cublas nanoflann::nanoflann)

add_executable(eval-knn-runtime eval_knn_runtime.cpp)
target_link_libraries(eval-knn-runtime PRIVATE ${DEPS})
