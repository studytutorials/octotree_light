cmake_minimum_required(VERSION 3.9...3.16)

add_executable(image-unittest "image_unittest.cpp")
gtest_add_tests(image-unittest "" AUTO)

