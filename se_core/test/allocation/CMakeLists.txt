cmake_minimum_required(VERSION 3.9...3.16)

add_executable(allocation-unittest "alloc_unittest.cpp")
gtest_add_tests(allocation-unittest "" AUTO)

