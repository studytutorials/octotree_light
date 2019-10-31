cmake_minimum_required(VERSION 3.10)

# Define the library targets. Use generator expressions (BUILD_INTERFACE and
# INSTALL_INTERFACE) to change the include directories when building the
# library and when it is installed and used from another project.

# Header only voxel implementation library.
add_library(se-voxel-impl INTERFACE)
target_include_directories(se-voxel-impl INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
install(
  TARGETS se-voxel-impl
  EXPORT supereightTargets)
# Populate the list of libraries that are part of se-voxel-impl.
list(APPEND SUPEREIGHT_CORE_LIBS se-voxel-impl)



# Install the headers.
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION include)

# When linking the se-voxel-impl library, the se-shared libraries should also
# be linked. Append them to the list.
list(APPEND SUPEREIGHT_CORE_LIBS ${SUPEREIGHT_SHARED_LIBS})

# Make the list of se-voxel-impl libraries available to the parent file.
set(SUPEREIGHT_CORE_LIBS ${SUPEREIGHT_CORE_LIBS} PARENT_SCOPE)

