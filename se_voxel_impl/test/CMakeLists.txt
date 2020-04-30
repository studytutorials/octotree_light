cmake_minimum_required(VERSION 3.10)
project(se_denseslam_unit_testing)

# Build the dependencies inside the test build directory.
add_subdirectory(../../third_party third_party)
add_subdirectory(../../se_shared se_shared)
add_subdirectory(../../se_core se_core)
add_subdirectory(../../se_voxel_impl se_voxel_impl)

# Add local path for finding packages.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

find_package(GTest REQUIRED)
find_package(Eigen3 REQUIRED)

enable_testing()
set(CMAKE_CXX_STANDARD 14)
include_directories(
	../../se_core/include
	../../se_shared/include
    ../include/se
	../include
    ${EIGEN3_INCLUDE_DIR}
)

add_definitions(-DSE_SENSOR_IMPLEMENTATION=PinholeCamera)

# Build the tests.
add_subdirectory(multires_esdf_moving_sphere)
add_subdirectory(multires_tsdf_moving_camera)

