cmake_minimum_required(VERSION 3.10)

# This project name will also be the prefix of the libraries compiled from
# se_denseslam.
project(se-denseslam)



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
set(libraries ${SUPEREIGHT_CORE_LIBS})
if(APPLE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D__APPLE__")
endif(APPLE)
if (WITH_OPENMP AND OPENMP_FOUND)
  list(APPEND compile_flags ${OpenMP_CXX_FLAGS})
  list(APPEND libraries ${OpenMP_CXX_FLAGS})
endif()



# Generate the appropriate include line for each voxel implementation.
foreach(voxel_impl ${SUPEREIGHT_VOXEL_IMPLS})
  string(APPEND
    voxel_impl_includes
    "#include <se/voxel_implementations/${voxel_impl}/${voxel_impl}.hpp>\n")
endforeach()

# Generate header file including each voxel implementation.
configure_file(
	"${PROJECT_SOURCE_DIR}/include/se/voxel_implementations.hpp.in"
	"${PROJECT_BINARY_DIR}/include/se/voxel_implementations.hpp"
)



# The source files to compile
set(se_src
  "./src/preprocessing.cpp"
  "./src/tracking.cpp"
  "./src/rendering.cpp"
  "./src/DenseSLAMSystem.cpp")



# Iterate over each SE_VOXEL_IMPLEMENTATION and compile the se-denseslam
# library for it.
foreach(voxel_impl ${SUPEREIGHT_VOXEL_IMPLS})
  # Convert the voxel implementation name to lowercase for use in filenames.
  string(TOLOWER ${voxel_impl} voxel_impl_lc)

  # Prepare the preprocessor SE_VOXEL_IMPLEMENTATION macro.
  set(field_type_macro SE_VOXEL_IMPLEMENTATION=${voxel_impl})

  # Prepare the library name.
  set(lib_name "${PROJECT_NAME}-${voxel_impl_lc}")

  # Find the name of the respective voxel-impl library.
  set(voxel_impl_lib_name "se-voxel-impl-${voxel_impl_lc}")

  # Add the static library.
  add_library(${lib_name} STATIC
    ${se_src})
  target_include_directories(${lib_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/se>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR})
  target_compile_options(${lib_name} PUBLIC ${compile_flags})
  target_link_libraries(${lib_name} ${libraries} ${voxel_impl_lib_name})
  target_compile_definitions(${lib_name} PUBLIC ${field_type_macro})

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
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
    DESTINATION include)

  # Populate the list of libraries that are part of se-denseslam.
  list(APPEND SUPEREIGHT_DENSESLAM_LIBS ${lib_name})
endforeach()



# Make the list of compiled se-denseslam libraries available to the parent file.
set(SUPEREIGHT_DENSESLAM_LIBS ${SUPEREIGHT_DENSESLAM_LIBS} PARENT_SCOPE)

