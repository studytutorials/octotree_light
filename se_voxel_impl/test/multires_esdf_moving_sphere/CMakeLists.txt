cmake_minimum_required(VERSION 3.9...3.16)

set(unit_test_name esdf-moving-sphere-unittest)
add_executable(${unit_test_name} "esdf_moving_sphere_unittest.cpp")
target_include_directories(${unit_test_name} BEFORE PRIVATE "../../include")
target_compile_definitions(${unit_test_name}
  PUBLIC
    SE_SENSOR_IMPLEMENTATION=PinholeCamera
)
gtest_add_tests(${unit_test_name} "" AUTO)

