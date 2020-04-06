# This CMake configuration is based on
# https://github.com/pabloariasal/modern-cmake-sample

cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
# Adding the C language is needed for CMake 3.0.0 compatibility
project(se-shared LANGUAGES C CXX)

# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# Dependencies
find_package(Eigen3 REQUIRED)
find_package(LodePNG REQUIRED)
find_package(SRLProjection REQUIRED)

##############################################
# Create library target
add_library(${PROJECT_NAME} STATIC
  src/image_utils.cpp
  src/sensor.cpp
  src/str_utils.cpp
)

# Add an alias so that the library can be used inside the build tree, e.g.
# when testing
add_library(SE::Shared ALIAS ${PROJECT_NAME})

# Set target properties
target_include_directories(${PROJECT_NAME} BEFORE
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    LodePNG::lodepng
    SRL::Projection
)

##############################################
# Setup installation

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/SE)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# This is required so that the exported target has the name Shared and not
# se-shared
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME Shared)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the targets to a script
install(EXPORT ${PROJECT_NAME}-targets
  FILE
    SESharedTargets.cmake
  NAMESPACE
    SE::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
# Generate a Config.cmake file
configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/SESharedConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/SESharedConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install the Config file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/SRLProjectionConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

##############################################
# Export from the build tree
export(EXPORT ${PROJECT_NAME}-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/SESharedTargets.cmake
  NAMESPACE SE::
)

# Register the package in the user's package registry
export(PACKAGE SEShared)

