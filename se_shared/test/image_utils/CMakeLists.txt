cmake_minimum_required(VERSION 2.10)
project(se_shared)

set(PROJECT_TEST_NAME image_utils)
add_executable(${PROJECT_TEST_NAME} "image_utils_unittest.cpp")
target_include_directories(${PROJECT_TEST_NAME} PUBLIC ${GTEST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES} pthread lodepng)

GTEST_ADD_TESTS(${PROJECT_TEST_NAME} "" AUTO)

