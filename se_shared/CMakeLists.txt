cmake_minimum_required(VERSION 3.5...3.16)

find_package(yaml-cpp 0.5.2 REQUIRED)

# Create library ##############################################################
set(LIB_NAME "se-shared")
add_library(${LIB_NAME} STATIC
  "src/image_utils.cpp"
  "src/sensor.cpp"
  "src/str_utils.cpp"
  "src/point_cloud_utils.cpp"
  "src/stats.cpp"
)
target_include_directories(${LIB_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${LIB_NAME}
  PUBLIC
    SE::Core
    SRL::Projection
	Lode::PNG
    ${YAML_CPP_LIBRARIES}
)
set_target_properties(${LIB_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)

# Add an alias so that the library can be used inside the build tree, e.g. when
# testing
add_library(SE::Shared ALIAS ${LIB_NAME})
# This is required so that the exported target has the name Shared and not
# se-shared
set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME Shared)



# Test library ################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()



# Install the library #########################################################
set(SUBPACKAGE_NAME SEShared)
# Export the targets to a script
install(EXPORT ${LIB_NAME}-targets
  FILE
    ${SUBPACKAGE_NAME}Targets.cmake
  NAMESPACE
    SE::
  DESTINATION
    "${CMAKE_INSTALL_CMAKEDIR}"
)
# Install the headers
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# Install the static library
install(TARGETS ${LIB_NAME}
  EXPORT ${LIB_NAME}-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
)



# Export from the build tree ##################################################
export(EXPORT ${LIB_NAME}-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBPACKAGE_NAME}Targets.cmake"
    NAMESPACE SE::
)

