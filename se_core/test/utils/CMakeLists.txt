cmake_minimum_required(VERSION 3.9...3.16)

add_executable(morton-unittest "morton_unittest.cpp")
gtest_add_tests(morton-unittest "" AUTO)

add_executable(math-unittest "math_unittest.cpp")
gtest_add_tests(math-unittest "" AUTO)

