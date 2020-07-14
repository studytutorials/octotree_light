cmake_minimum_required(VERSION 3.9...3.16)

add_executable(image-utils-unittest "image_utils_unittest.cpp")
gtest_add_tests(image-utils-unittest "" AUTO)

