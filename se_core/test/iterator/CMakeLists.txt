cmake_minimum_required(VERSION 3.9...3.16)

add_executable(iterator-unittest "iterator_unittest.cpp")
gtest_add_tests(iterator-unittest "" AUTO)

