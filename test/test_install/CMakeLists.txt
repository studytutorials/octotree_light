cmake_minimum_required(VERSION 3.5...3.16)

project(se-install-test VERSION 0.1.0 LANGUAGES C CXX)

enable_testing()



find_package(LodePNG REQUIRED)

# LodePNG
set(EXE_NAME "lodepng-install-test")
add_executable(${EXE_NAME} "lodepng_install_test.cpp")
target_link_libraries(${EXE_NAME} PRIVATE Lode::PNG)
add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})



find_package(supereight REQUIRED)

# se-core
set(EXE_NAME "se-core-install-test")
add_executable(${EXE_NAME} "se_core_install_test.cpp")
target_link_libraries(${EXE_NAME} PRIVATE SE::Core)
add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})

# se-shared
set(EXE_NAME "se-shared-install-test")
add_executable(${EXE_NAME} "se_shared_install_test.cpp")
target_link_libraries(${EXE_NAME} PRIVATE SE::Shared)
add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})

# se-denseslam
set(EXE_NAME "se-denseslam-install-test")
add_executable(${EXE_NAME} "se_denseslam_install_test.cpp")
target_link_libraries(${EXE_NAME} PRIVATE SE::DenseSLAMTSDFPinholeCamera)
add_test(NAME ${EXE_NAME} COMMAND ${EXE_NAME})

