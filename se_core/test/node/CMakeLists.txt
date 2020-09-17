cmake_minimum_required(VERSION 3.9...3.16)

add_executable(node-unittest "node_unittest.cpp")
gtest_add_tests(node-unittest "" AUTO)

add_executable(voxelblock-common-unittest "voxelblock_common_unittest.cpp")
gtest_add_tests(voxelblock-common-unittest "" AUTO)

