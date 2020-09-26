cmake_minimum_required(VERSION 3.5...3.16)

# Disable warnings for third-party code
add_compile_options(-w)

# googletest / GTEST
if (SE_BUILD_TESTS)
  option(BUILD_GMOCK "Builds the googlemock subproject" OFF)
  option(INSTALL_GTEST "Enable installation of googletest. (Projects embedding googletest may want to turn this OFF.)" OFF)
  add_subdirectory(googletest)
endif()

# LodePNG
add_subdirectory(lodepng)

# YAML-CPP
option(YAML_CPP_BUILD_TESTS "Enable testing" OFF)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" OFF)
add_subdirectory(yaml-cpp)

# SRL Projection
option(SRL_PROJECTION_BUILD_TESTS "Enable testing" OFF)
add_subdirectory(srl_projection)

