cmake_minimum_required(VERSION 3.9...3.16)

add_executable(gather-unittest "gather_unittest.cpp")
gtest_add_tests(gather-unittest "" AUTO)

add_executable(interpolation-unittest "interpolation_unittest.cpp")
gtest_add_tests(interpolation-unittest "" AUTO)

