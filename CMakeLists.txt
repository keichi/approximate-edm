cmake_minimum_required(VERSION 3.16)
project(approx-edm CXX)

enable_language(CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(faiss REQUIRED)

add_subdirectory(thirdparty/cxxopts)
add_subdirectory(thirdparty/xtl)
add_subdirectory(thirdparty/xtensor)
add_subdirectory(thirdparty/nanoflann)

set(DEPS OpenMP::OpenMP_CXX xtl xtensor faiss CUDA::cublas
    nanoflann::nanoflann cxxopts::cxxopts)

add_executable(eval-knn-runtime eval_knn_runtime.cpp)
target_link_libraries(eval-knn-runtime PRIVATE ${DEPS})
