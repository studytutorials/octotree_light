cmake_minimum_required(VERSION 3.5...3.16)

find_package(Sophus 1.0 REQUIRED)
find_package(octomap)

# Generate the appropriate include line for each voxel implementation.
foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
  string(APPEND VOXEL_IMPL_INCLUDES
    "#include <se/voxel_implementations/${VOXEL_IMPL}/${VOXEL_IMPL}.hpp>\n")
endforeach()

# Generate header file including each voxel implementation.
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/se/voxel_implementations.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/se/voxel_implementations.hpp"
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/include/se/voxel_implementations.hpp"
  DESTINATION include/se
)



# Create library ##############################################################
# Iterate over each voxel and sensor implementaion and compile into an
# se-denseslam library.
foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
  foreach(SENSOR_IMPL ${SE_SENSOR_IMPLS})
    # Prepare the library name
    string(TOLOWER ${VOXEL_IMPL} VOXEL_IMPL_LC)
    string(TOLOWER ${SENSOR_IMPL} SENSOR_IMPL_LC)
    set(LIB_NAME "se-denseslam-${VOXEL_IMPL_LC}-${SENSOR_IMPL_LC}")

    # Add the static library
    add_library(${LIB_NAME} STATIC
      "./src/preprocessing.cpp"
      "./src/tracking.cpp"
      "./src/rendering.cpp"
      "./src/DenseSLAMSystem.cpp"
    )
    target_include_directories(${LIB_NAME} BEFORE
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${LIB_NAME}
      PUBLIC
        Sophus::Sophus
        SE::VoxelImpl${VOXEL_IMPL}${SENSOR_IMPL}
    )
    set_target_properties(${LIB_NAME} PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    if(DEFINED OCTOMAP_INCLUDE_DIRS)
      target_include_directories(${LIB_NAME} PUBLIC ${OCTOMAP_INCLUDE_DIRS})
      target_link_libraries(${LIB_NAME} PUBLIC ${OCTOMAP_LIBRARIES})
      target_compile_definitions(${LIB_NAME} PUBLIC SE_OCTOMAP=1)
    else()
      target_compile_definitions(${LIB_NAME} PUBLIC SE_OCTOMAP=0)
    endif()

    # Add an alias so that the library can be used inside the build tree, e.g.
    # when testing
    add_library(SE::DenseSLAM${VOXEL_IMPL}${SENSOR_IMPL} ALIAS ${LIB_NAME})
    # This is required so that the exported target has the name
    # DenseSLAMVoxelSensor and not se-denseslam-voxel-sensor
    set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME DenseSLAM${VOXEL_IMPL}${SENSOR_IMPL})



    # Install library #########################################################
    set(SUBPACKAGE_NAME SEDenseSLAM${VOXEL_IMPL}${SENSOR_IMPL})
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

    # Export from the build tree ##############################################
    export(EXPORT ${LIB_NAME}-targets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBPACKAGE_NAME}Targets.cmake"
        NAMESPACE SE::
    )
  endforeach()
endforeach()



# Test library ################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()

