cmake_minimum_required(VERSION 3.10)
project(se_core_unit_testing)

# Add local path for finding packages.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()
set(CMAKE_CXX_STANDARD 14)
include_directories(
	../include/se
	../include
	${EIGEN3_INCLUDE_DIR}
)

add_subdirectory(octree)
add_subdirectory(io)
add_subdirectory(allocation)
add_subdirectory(functor)
add_subdirectory(interp)
add_subdirectory(multiscale)
add_subdirectory(algorithms)
add_subdirectory(geometry)
add_subdirectory(utils)
add_subdirectory(neighbors)
