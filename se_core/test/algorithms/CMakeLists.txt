cmake_minimum_required(VERSION 3.9...3.16)

add_executable(unique-unittest "unique_unittest.cpp")
gtest_add_tests(unique-unittest "" AUTO)

add_executable(meshing-unittest "meshing_unittest.cpp")
gtest_add_tests(meshing-unittest "" AUTO)

