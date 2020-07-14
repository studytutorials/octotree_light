cmake_minimum_required(VERSION 3.5...3.16)

# Create library ##############################################################
# Iterate over each voxel and sensor implementaion and compile into an
# se-voxel-impl library.
foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
  foreach(SENSOR_IMPL ${SE_SENSOR_IMPLS})
    # Prepare the library name
    string(TOLOWER ${VOXEL_IMPL} VOXEL_IMPL_LC)
    string(TOLOWER ${SENSOR_IMPL} SENSOR_IMPL_LC)
    set(LIB_NAME "se-voxel-impl-${VOXEL_IMPL_LC}-${SENSOR_IMPL_LC}")

    # Find the source files to compile
    file(GLOB LIB_SRC "./src/${VOXEL_IMPL}/*.cpp")

    # Add the static library
    add_library(${LIB_NAME} STATIC ${LIB_SRC})
    target_include_directories(${LIB_NAME} BEFORE
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(${LIB_NAME}
      PUBLIC
        SE::Shared
    )
    set_target_properties(${LIB_NAME} PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
	# Define the voxel and sensor implementation macros
    target_compile_definitions(${LIB_NAME}
      PUBLIC
        SE_VOXEL_IMPLEMENTATION=${VOXEL_IMPL}
        SE_SENSOR_IMPLEMENTATION=${SENSOR_IMPL}
    )

    # Add an alias so that the library can be used inside the build tree, e.g.
    # when testing
    add_library(SE::VoxelImpl${VOXEL_IMPL}${SENSOR_IMPL} ALIAS ${LIB_NAME})
    # This is required so that the exported target has the name
    # VoxelImplVoxelSensor and not se-voxel-impl-voxel-sensor
    set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME VoxelImpl${VOXEL_IMPL}${SENSOR_IMPL})



    # Install library #########################################################
    set(SUBPACKAGE_NAME SEVoxelImpl${VOXEL_IMPL}${SENSOR_IMPL})
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

