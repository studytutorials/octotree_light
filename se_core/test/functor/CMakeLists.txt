cmake_minimum_required(VERSION 3.9...3.16)

add_executable(aa-functor-unittest "axisaligned_unittest.cpp")
gtest_add_tests(aa-functor-unittest "" AUTO)

