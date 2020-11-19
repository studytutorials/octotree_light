cmake_minimum_required(VERSION 3.9...3.16)

add_executable(image-utils-unittest "image_utils_unittest.cpp")
gtest_add_tests(image-utils-unittest "" AUTO)

add_executable(image-utils-io-unittest "image_utils_io_unittest.cpp")
gtest_add_tests(image-utils-io-unittest "" AUTO)

