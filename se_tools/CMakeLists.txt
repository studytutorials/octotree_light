cmake_minimum_required(VERSION 3.5...3.16)

find_package(OpenNI2)
if (OPENNI2_FOUND)
  add_executable(oni2raw "oni2raw.cpp")
  target_include_directories(oni2raw BEFORE
    PRIVATE
      ${OPENNI2_INCLUDE_PATHS}
  )
  target_link_libraries(oni2raw
    PRIVATE
      ${OPENNI2_LIBRARIES}
  )
endif(OPENNI2_FOUND)

add_executable(scene2raw scene2raw.cpp)
target_link_libraries(scene2raw PRIVATE Lode::PNG)

