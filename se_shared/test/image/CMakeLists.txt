cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
project(image-test LANGUAGES C CXX)

find_package(Eigen3 REQUIRED)

add_executable(${PROJECT_NAME}
  image_unittest.cpp
)
target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIRS}
)
target_link_libraries(${PROJECT_NAME}
  SE::Shared
  ${GTEST_BOTH_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

