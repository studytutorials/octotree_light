cmake_minimum_required(VERSION 3.10)
project(se_shared_unit_testing)

# Add local path for finding packages.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()
set(CMAKE_CXX_STANDARD 14)
include_directories(
  ../include/se
  ../include
  ${EIGEN3_INCLUDE_DIR})

# Build the dependencies inside the test build directory.
add_subdirectory(../../se_shared se_shared)

# Build the tests.
add_subdirectory(image_utils)

