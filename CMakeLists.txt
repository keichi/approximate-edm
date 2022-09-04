cmake_minimum_required(VERSION 3.16)
project(approx-edm CXX)

add_executable(approx-edm approx_edm.cpp)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
target_include_directories(approx-edm PUBLIC ${xtl_INCLUDE_DIRS})
target_link_libraries(approx-edm PUBLIC xtl xtensor)

find_package(OpenMP REQUIRED)
find_package(faiss REQUIRED)
target_link_libraries(approx-edm PUBLIC faiss)
