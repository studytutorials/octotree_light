cmake_minimum_required(VERSION 3.9...3.16)

add_executable(neighbor-gather-unittest "neighbor_gather_unittest.cpp")
gtest_add_tests(neighbor-gather-unittest "" AUTO)

