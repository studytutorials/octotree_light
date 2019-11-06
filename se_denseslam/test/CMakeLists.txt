cmake_minimum_required(VERSION 3.10)
project(se_denseslam_unit_testing)

# Add local path for finding packages.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)

enable_testing()
set(CMAKE_CXX_STANDARD 14)
include_directories(
	../../se_shared/include
	../../se_core/include
	../../se_voxel_impl/include
    ../include/se
	../include
    ${EIGEN3_INCLUDE_DIR}
    ${SOPHUS_INCLUDE_DIR}
)

# Build the dependencies inside the test build directory.
add_subdirectory(../../se_shared se_shared)
add_subdirectory(../../se_core se_core)
add_subdirectory(../../se_voxel_impl se_voxel_impl)

# Build the tests.
add_subdirectory(preprocessing)
