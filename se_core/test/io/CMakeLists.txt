cmake_minimum_required(VERSION 3.9...3.16)

add_executable(octree-io-unittest "io_unittest.cpp")
gtest_add_tests(octree-io-unittest "" AUTO)

set(unit_test_name octomap-io-unittest)
add_executable(${unit_test_name} "octomap_io_unittest.cpp")
find_package(octomap)
if(DEFINED OCTOMAP_INCLUDE_DIRS)
  target_include_directories(${unit_test_name} PUBLIC ${OCTOMAP_INCLUDE_DIRS})
  target_link_libraries(${unit_test_name} ${OCTOMAP_LIBRARIES})
  target_compile_definitions(${unit_test_name} PUBLIC SE_OCTOMAP=1)
else()
  target_compile_definitions(${unit_test_name} PUBLIC SE_OCTOMAP=0)
endif()
gtest_add_tests(${unit_test_name} "" AUTO)

