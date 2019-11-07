cmake_minimum_required(VERSION 3.10)

# This project name will also be the prefix of the libraries compiled from
# se_voxel_impl.
project(se-voxel-impl)



# Find dependencies.
find_package(Eigen3 REQUIRED)
find_package(Sophus REQUIRED)
if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "Compiling with OpenMP support")
  endif()
endif()



# Set the compiler flags and linked libraries.
set(compile_flags -Wall -Wextra -Wno-unknown-pragmas)
set(libraries se-shared se-core)
if(APPLE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__APPLE__")
endif(APPLE)
if(WITH_OPENMP AND OPENMP_FOUND)
  list(APPEND compile_flags ${OpenMP_CXX_FLAGS})
  list(APPEND libraries ${OpenMP_CXX_FLAGS})
endif()



# List of voxel implementations to compile. The list elements must correspond
# to folder names inside se_voxel_impl/include/se/voxel_implementations. When
# adding a new voxel implementation, appending it to this list is enough to
# compile supereight with it.
set(SUPEREIGHT_VOXEL_IMPLS TSDF OFusion MultiresTSDF)

# Iterate over each voxel implementaion and compile into an se-voxel-impl
# library.
foreach(voxel_impl ${SUPEREIGHT_VOXEL_IMPLS})
  # Convert the voxel implementation name to lowercase for use in filenames.
  string(TOLOWER ${voxel_impl} voxel_impl_lc)

  # Prepare the library name.
  set(lib_name "${PROJECT_NAME}-${voxel_impl_lc}")

  # Find the source files to compile.
  file(GLOB lib_src "./src/${voxel_impl}/*.cpp")

  # Add the static library.
  add_library(${lib_name} STATIC
    ${lib_src})
  target_include_directories(${lib_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR})
  target_compile_options(${lib_name} PUBLIC ${compile_flags})
  target_link_libraries(${lib_name} ${libraries})

  # Install the library and headers.
  install(
    TARGETS ${lib_name}
    EXPORT supereightTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include)

  # Populate the list of libraries that are part of se-voxel-impl.
  list(APPEND SUPEREIGHT_VOXEL_IMPL_LIBS ${lib_name})
endforeach()

# Make the list of voxel implementations available to the parent file.
set(SUPEREIGHT_VOXEL_IMPLS ${SUPEREIGHT_VOXEL_IMPLS} PARENT_SCOPE)

# Make the list of compiled se-voxel-impl libraries available to the parent
# file.
set(SUPEREIGHT_VOXEL_IMPL_LIBS ${SUPEREIGHT_VOXEL_IMPL_LIBS} PARENT_SCOPE)

