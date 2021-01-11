cmake_minimum_required(VERSION 3.5...3.16)

find_package(OpenNI2)
find_package(GLUT)
find_package(OpenGL)
find_package(Qt5 COMPONENTS OpenGL PrintSupport Widgets)
if(Qt5_FOUND)
  set(CMAKE_AUTOMOC ON)
  set(CMAKE_AUTORCC ON)
  set(CMAKE_AUTOUIC ON)
  if(CMAKE_VERSION VERSION_LESS "3.7.0")
    set(CMAKE_INCLUDE_CURRENT_DIR ON)
  endif()
endif()

set(READER_SRC
  src/reader_base.cpp
  src/reader_iclnuim.cpp
  src/reader_newercollege.cpp
  src/reader_openni.cpp
  src/reader_raw.cpp
  src/reader.cpp
)

# Create executables ##########################################################
# Iterate over each voxel and sensor implementaion and compile the executables
# for it.
foreach(VOXEL_IMPL ${SE_VOXEL_IMPLS})
  foreach(SENSOR_IMPL ${SE_SENSOR_IMPLS})
    # Convert the implementation names to lowercase
    string(TOLOWER ${VOXEL_IMPL} VOXEL_IMPL_LC)
    string(TOLOWER ${SENSOR_IMPL} SENSOR_IMPL_LC)

    # Main executable #########################################################
    set(EXE_NAME "se-denseslam-${VOXEL_IMPL_LC}-${SENSOR_IMPL_LC}-main")
    add_executable(${EXE_NAME}
      "src/main.cpp"
      "src/PowerMonitor.cpp"
      ${READER_SRC}
    )
    target_include_directories(${EXE_NAME} BEFORE PRIVATE include)
    target_link_libraries(${EXE_NAME}
      PRIVATE
        SE::DenseSLAM${VOXEL_IMPL}${SENSOR_IMPL}
        stdc++fs
    )
    # Allow handling large files in 32-bit systems
    target_compile_definitions(${EXE_NAME} PRIVATE "-D_FILE_OFFSET_BITS=64")
    set_target_properties(${EXE_NAME} PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
    )
    # Compile with GUI support
    if(SE_BUILD_GLUT_GUI AND GLUT_FOUND)
      target_link_libraries(${EXE_NAME}
        PRIVATE
          ${GLUT_LIBRARIES}
          ${OPENGL_LIBRARIES}
      )
      target_compile_definitions(${EXE_NAME} PRIVATE "-DSE_GLUT")
      if(NOT APPLE)
        target_link_libraries(${EXE_NAME} PRIVATE GL)
      endif()
    endif()
    # Compile with OpenNI2 support
    if(OPENNI2_FOUND)
      target_include_directories(${EXE_NAME} PRIVATE ${OPENNI2_INCLUDE_PATHS})
      target_link_libraries(${EXE_NAME} PRIVATE ${OPENNI2_LIBRARIES})
      target_compile_definitions(${EXE_NAME} PRIVATE "-DSE_USE_OPENNI")
    endif()



    # Qt executable ###########################################################
    if(SE_BUILD_QT_GUI AND Qt5_FOUND)
      # Find the Qt source files
      file(GLOB QT_SRC qt/*)
      set(QT_SRC ${QT_SRC} "qt/images.qrc")

      set(EXE_NAME "se-denseslam-${VOXEL_IMPL_LC}-${SENSOR_IMPL_LC}-qt")
      add_executable(${EXE_NAME}
        "src/main.cpp"
        "src/PowerMonitor.cpp"
        ${READER_SRC}
        ${QT_SRC}
      )
      target_include_directories(${EXE_NAME} BEFORE PRIVATE include qt)
      target_link_libraries(${EXE_NAME}
        PRIVATE
          SE::DenseSLAM${VOXEL_IMPL}${SENSOR_IMPL}
          stdc++fs
          Qt5::OpenGL
          Qt5::PrintSupport
          Qt5::Widgets
      )
      target_compile_definitions(${EXE_NAME}
        PRIVATE
          "-DSE_QT"
          "-D_FILE_OFFSET_BITS=64"
      )
      set_target_properties(${EXE_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
      )
      if(NOT APPLE)
        target_link_libraries(${EXE_NAME} PRIVATE GL)
      endif()
      # Compile with OpenNI2 support
      if(OPENNI2_FOUND)
        target_include_directories(${EXE_NAME} PRIVATE ${OPENNI2_INCLUDE_PATHS})
        target_link_libraries(${EXE_NAME} PRIVATE ${OPENNI2_LIBRARIES})
        target_compile_definitions(${EXE_NAME} PRIVATE "-DSE_USE_OPENNI")
      endif()
    endif()
  endforeach()
endforeach()



# Executable showing voxel data sizes
set(EXE_NAME "se-voxel-size")
add_executable(${EXE_NAME} "src/voxel_size.cpp")
target_include_directories(${EXE_NAME} BEFORE PRIVATE ../se_voxel_impl/include)
target_link_libraries(${EXE_NAME} PRIVATE SE::Shared)
target_compile_definitions(${EXE_NAME} PUBLIC SE_SENSOR_IMPLEMENTATION=PinholeCamera)



# Test ########################################################################
if(SE_BUILD_TESTS)
  add_subdirectory(test)
endif()

