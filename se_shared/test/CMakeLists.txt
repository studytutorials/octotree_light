cmake_minimum_required(VERSION 3.5...3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Include directories common for all tests
include_directories(${GTEST_INCLUDE_DIRS})

# Libraries to link all tests with
link_libraries(
  SE::Shared
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

# Preprocessor definitions for all tests
add_definitions(-DSE_SENSOR_IMPLEMENTATION=PinholeCamera)

add_subdirectory(image)
add_subdirectory(image_utils)

