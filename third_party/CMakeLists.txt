cmake_minimum_required(VERSION 3.5...3.16)

# Disable warnings for third-party code
add_compile_options(-w)

# LodePNG
add_subdirectory(lodepng)

# Sophus
option(BUILD_TESTS "Build tests." OFF)
option(BUILD_EXAMPLES "Build examples." OFF)
add_subdirectory(sophus)

# YAML-CPP
option(YAML_CPP_BUILD_TESTS "Enable testing" OFF)
option(YAML_CPP_BUILD_TOOLS "Enable parse tools" OFF)
add_subdirectory(yaml-cpp)

# SRL Projection
option(SRL_PROJECTION_BUILD_TESTS "Enable testing" OFF)
add_subdirectory(srl_projection)

