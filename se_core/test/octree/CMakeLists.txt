cmake_minimum_required(VERSION 3.9...3.16)

add_executable(octree-unittest "octree_unittest.cpp")
gtest_add_tests(octree-unittest "" AUTO)

add_executable(voxelblock-ray-iterator-unittest "voxel_block_ray_iterator_unittest.cpp")
gtest_add_tests(voxelblock-ray-iterator-unittest "" AUTO)

add_executable(voxelblock-unittest "voxel_block_unittest.cpp")
gtest_add_tests(voxelblock-unittest "" AUTO)

