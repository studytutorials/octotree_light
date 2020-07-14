cmake_minimum_required(VERSION 3.9...3.16)

add_executable(reader-raw-unittest
  "../../src/reader_base.cpp"
  "../../src/reader_raw.cpp"
  "reader_raw_unittest.cpp"
)
gtest_add_tests(reader-raw-unittest "" AUTO)

