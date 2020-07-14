cmake_minimum_required(VERSION 3.5...3.16)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

# Create library ##############################################################
set(LIB_NAME "se-core")
add_library(${LIB_NAME} INTERFACE)
target_include_directories(${LIB_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(${LIB_NAME}
  INTERFACE
    Eigen3::Eigen
)

# Add an alias so that the library can be used inside the build tree, e.g. when
# testing
add_library(SE::Core ALIAS ${LIB_NAME})
# This is required so that the exported target has the name Core and not
# se-core
set_target_properties(${LIB_NAME} PROPERTIES EXPORT_NAME Core)



# Test library ################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()



# Install the library #########################################################
set(SUBPACKAGE_NAME SECore)
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
install(TARGETS ${LIB_NAME}
  EXPORT ${LIB_NAME}-targets
)



# Export from the build tree ##################################################
export(EXPORT ${LIB_NAME}-targets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBPACKAGE_NAME}Targets.cmake"
    NAMESPACE SE::
)

