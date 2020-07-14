cmake_minimum_required(VERSION 3.9...3.16)

add_executable(aabb-collision-unittest "aabb_collision_unittest.cpp")
gtest_add_tests(aabb-collision-unittest "" AUTO)

add_executable(octree-collision-unittest "octree_collision_unittest.cpp")
gtest_add_tests(octree-collision-unittest "" AUTO)

