/*
 * SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

// Ensure OpenNI won't be used
#ifdef SE_USE_OPENNI
#undef SE_USE_OPENNI
#endif

#include <cstdint>
#include <string>

#include "reader_openni.hpp"



TEST(NoOpenNI, initAndRead) {
  // Initialize the reader.
  const se::ReaderConfig config_ = {0, false, "", ""};
  se::OpenNIReader reader (config_);

  // Test for correct initial state.
  ASSERT_EQ(reader.name(), "OpenNIReader");
  ASSERT_FALSE(reader.good());
  ASSERT_EQ(reader.frame(), SIZE_MAX);
  ASSERT_EQ(reader.depthImageRes(), Eigen::Vector2i::Ones());
  ASSERT_EQ(reader.RGBAImageRes(),  Eigen::Vector2i::Ones());

  // Restart and test for correct state.
  reader.restart();
  ASSERT_FALSE(reader.good());
  ASSERT_EQ(reader.frame(), SIZE_MAX);
}

