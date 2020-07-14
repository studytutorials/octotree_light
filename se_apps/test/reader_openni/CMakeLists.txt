cmake_minimum_required(VERSION 3.9...3.16)

add_executable(reader-openni-unittest
  "../../src/reader_base.cpp"
  "../../src/reader_openni.cpp"
  "reader_no_openni_unittest.cpp"
)
gtest_add_tests(reader-openni-unittest "" AUTO)

