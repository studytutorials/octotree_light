cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(se-shared-tests LANGUAGES C CXX)

# Build the dependencies inside the test build directory.
add_subdirectory(../../third_party third_party)
add_subdirectory(../../se_shared se_shared)

# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# Testing dependencies
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(SEShared REQUIRED)

add_definitions(-DSE_SENSOR_IMPLEMENTATION=PinholeCamera)

enable_testing()

# Build the tests
add_subdirectory(image)
add_subdirectory(image_utils)

