# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
# SPDX-License-Identifier: CC0-1.0

cmake_minimum_required(VERSION 3.9.0 FATAL_ERROR)
project(octree-io-unittest LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)
find_package(octomap)

add_executable(${PROJECT_NAME}
  io_unittest.cpp
  octomap_io_unittest.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ../../include
    ${EIGEN3_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

if(DEFINED OCTOMAP_INCLUDE_DIRS)
  target_include_directories(${PROJECT_NAME} PUBLIC ${OCTOMAP_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} ${OCTOMAP_LIBRARIES})
  target_compile_definitions(${PROJECT_NAME} PUBLIC SE_OCTOMAP=1)
else()
  target_compile_definitions(${PROJECT_NAME} PUBLIC SE_OCTOMAP=0)
endif()

gtest_add_tests(${PROJECT_NAME} "" AUTO)

