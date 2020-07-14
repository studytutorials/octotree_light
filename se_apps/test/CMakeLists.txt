cmake_minimum_required(VERSION 3.9...3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)

# Include directories common for all tests
include_directories("../include" ${GTEST_INCLUDE_DIRS})

# Libraries to link all tests with
link_libraries(
  SE::Shared
  stdc++fs
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

add_subdirectory(reader_iclnuim)
add_subdirectory(reader_openni)
add_subdirectory(reader_raw)

