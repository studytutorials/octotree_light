# This CMake configuration is based on
# https://github.com/pabloariasal/modern-cmake-sample

cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)
# Adding the C language is needed for CMake 3.0.0 compatibility
project(lodepng VERSION 2014.06.24 LANGUAGES C CXX)

# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

##############################################
# Create target and set properties

add_library(${PROJECT_NAME} STATIC
  src/lodepng.cpp
)

# Add an alias so that the library can be used inside the build tree, e.g. when
# testing
add_library(LodePNG::lodepng ALIAS ${PROJECT_NAME})

# Set target properties
target_include_directories(${PROJECT_NAME} BEFORE
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
  -Wall -pedantic -Wextra -Werror>
)

##############################################
# Installation instructions

include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/LodePNG)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# This is required so that the exported target has the name Projection and not
# srl-projection
set_target_properties(${PROJECT_NAME} PROPERTIES EXPORT_NAME lodepng)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export the targets to a script
install(EXPORT ${PROJECT_NAME}-targets
  FILE
    LodePNGTargets.cmake
  NAMESPACE
    LodePNG::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

# Create a ConfigVersion.cmake file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/LodePNGConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/LodePNGConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/LodePNGConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# Install the Config and ConfigVersion files
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/LodePNGConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/LodePNGConfigVersion.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

##############################################
## Exporting from the build tree
export(EXPORT ${PROJECT_NAME}-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/LodePNGTargets.cmake
  NAMESPACE LodePNG::
)

# Register package in user's package registry
export(PACKAGE LodePNG)

