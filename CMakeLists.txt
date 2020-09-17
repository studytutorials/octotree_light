cmake_minimum_required(VERSION 3.5...3.16)

project(supereight VERSION 0.1.0 LANGUAGES C CXX)

# The voxel implementations to compile. The list elements must correspond
# to folder names inside se_voxel_impl/include/se/voxel_implementations. When
# adding a new voxel implementation, appending it to this list is enough to
# compile supereight with it.
set(SE_VOXEL_IMPLS OFusion MultiresTSDF TSDF CACHE STRING "The voxel implementations to compile")

# The camera implementations to compile. The valid values are the names of the
# *Sensor classes defined in se_shared/include/se/sensor.hpp.
set(SE_SENSOR_IMPLS PinholeCamera OusterLidar CACHE STRING "The camera implementations to compile")

option(SE_BUILD_TESTS "Build the supereight unit tests" ON)
option(SE_BUILD_GLUT_GUI "Build the OpenGL-based GUI" ON)
option(SE_BUILD_QT_GUI "Build the Qt-based GUI" ON)
option(SE_USE_OMP "Compile with OpenMP" ON)

# Compile without measuring individual function times
add_definitions(-DSE_ENABLE_TICKTOCK=0)

# Compiler options
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wno-unknown-pragmas
  -Wno-unused-function
)



# Make sure that custom modules are found
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

# Set the build type to Release if not already set
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
# Warn if the release type is not Release
if(NOT "${CMAKE_BUILD_TYPE}" STREQUAL "Release")
  message(WARNING "CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}. Performance may be terrible.")
else()
  message(STATUS "CMAKE_BUILD_TYPE set to ${CMAKE_BUILD_TYPE}.")
endif()

# Find OpenMP, warn if disabled
if(SE_USE_OMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
    link_libraries(${OpenMP_CXX_LIBRARIES})
    message(STATUS "Compiling with OpenMP support")
  else()
    message(WARNING "OpenMP not found. Performance may be terrible.")
  endif()
else(NOT SE_USE_OMP)
  message(WARNING "Building without OpenMP. Performance may be terrible.")
endif()

# Installation includes
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)
# The directory where the .cmake files will be installed
if(WIN32 AND NOT CYGWIN)
  set(CMAKE_INSTALL_CMAKEDIR "${PROJECT_NAME}/cmake")
elseif(APPLE)
  set(CMAKE_INSTALL_CMAKEDIR "${PROJECT_NAME}.framework/Resources/CMake")
else()
  set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
endif()



# Enable tests
if(SE_BUILD_TESTS)
  enable_testing()
endif()

# Couldn't get Sophus to work as a submodule any other way
include_directories(third_party/sophus)
# Build each supereight module
add_subdirectory(third_party)
add_subdirectory(se_core)
add_subdirectory(se_shared)
add_subdirectory(se_voxel_impl)
add_subdirectory(se_denseslam)
add_subdirectory(se_apps)
add_subdirectory(se_tools)



# Install #####################################################################
# Create a Config.cmake file
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)

# Create a ConfigVersion.cmake file
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)



# Install the Config and ConfigVersion files
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_CMAKEDIR}"
)

# Create an uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
    IMMEDIATE @ONLY
  )
  add_custom_target(uninstall
    COMMAND
    ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/uninstall.cmake"
  )
endif()



# Export from the build tree #######################################################################
# Register package in user's package registry
export(PACKAGE ${PROJECT_NAME})

