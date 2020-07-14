cmake_minimum_required(VERSION 3.9...3.16)

add_executable(multiscale-unittest "multiscale_unittest.cpp")
gtest_add_tests(multiscale-unittest "" AUTO)

