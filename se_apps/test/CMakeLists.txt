cmake_minimum_required(VERSION 3.9...3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

include(GoogleTest)

# Include directories common for all tests
include_directories("../include")

# Libraries to link all tests with
link_libraries(
  SE::Shared
  stdc++fs
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)

add_subdirectory(reader_iclnuim)
add_subdirectory(reader_openni)
add_subdirectory(reader_raw)

