cmake_minimum_required(VERSION 3.5...3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

include(GoogleTest)

# Libraries to link all tests with
link_libraries(
  SE::Core
  gtest_main
  ${CMAKE_THREAD_LIBS_INIT}
)

add_subdirectory(algorithms)
add_subdirectory(allocation)
add_subdirectory(functor)
add_subdirectory(geometry)
add_subdirectory(interp)
add_subdirectory(io)
add_subdirectory(iterator)
add_subdirectory(multiscale)
add_subdirectory(neighbors)
add_subdirectory(node)
add_subdirectory(octree)
add_subdirectory(utils)

