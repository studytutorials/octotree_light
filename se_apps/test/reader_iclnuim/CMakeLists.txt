cmake_minimum_required(VERSION 3.9...3.16)

add_executable(reader-iclnuim-unittest
  "../../src/reader_base.cpp"
  "../../src/reader_iclnuim.cpp"
  "reader_iclnuim_unittest.cpp"
)
gtest_add_tests(reader-iclnuim-unittest "" AUTO)

