/*
 * SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>

#include "reader_raw.hpp"
#include "filesystem.hpp"



class TestRAWDataset : public ::testing::Test {
  protected:
    virtual void SetUp() {
      std::string tmpdir (stdfs::temp_directory_path());
      // Create a new .raw file in the temporary directory.
      raw_filename_ = tmpdir + "/test_raw.raw";
      std::ofstream fs (raw_filename_, std::ios::out | std::ios::binary);

      // Generate the .raw file.
      for (size_t frame = 0; frame < num_frames_; ++frame) {
        // Write the depth image.
        fs.write(reinterpret_cast<const char*>(&width_), sizeof(uint32_t));
        fs.write(reinterpret_cast<const char*>(&height_), sizeof(uint32_t));
        for (size_t pixel = 0; pixel < width_ * height_; ++pixel) {
          const uint16_t pixel_value = 1000 * (frame + pixel);
          fs.write(reinterpret_cast<const char*>(&pixel_value), sizeof(uint16_t));
        }

        // Write the RGBA image.
        fs.write(reinterpret_cast<const char*>(&width_), sizeof(uint32_t));
        fs.write(reinterpret_cast<const char*>(&height_), sizeof(uint32_t));
        for (size_t pixel = 0; pixel < width_ * height_; ++pixel) {
          const uint32_t pixel_value = frame + 10 * pixel;
          fs.write(reinterpret_cast<const char*>(&pixel_value), 3 * sizeof(uint8_t));
        }
      }
      fs.close();

      // Create a new groundtruth .txt file in the temporary directory.
      gt_filename_ = tmpdir + "/test_gt.txt";
      fs.open(gt_filename_, std::ios::out);
      // Generate the groundtruth .txt file.
      fs << std::showpoint;
      for (size_t frame = 0; frame < num_frames_; ++frame) {
        fs << "some garbage data "
           << frame << " " << frame + 1 << " " << frame + 2 << " "
           << 0.f << " " << 0.f << " " << 0.f << " " << 1.f << "\n";
      }
      fs.close();

      // Generate the reader config.
      config_ = {0, false, raw_filename_, gt_filename_};
    }

    const size_t num_frames_  = 2;
    const uint32_t width_     = 8;
    const uint32_t height_    = 4;
    std::string raw_filename_;
    std::string gt_filename_;
    se::ReaderConfig config_;
};



TEST_F(TestRAWDataset, ReadDepthRGBAPose) {
  // Initialize the reader.
  se::RAWReader reader (config_);

  // Test for correct initial state.
  ASSERT_EQ(reader.name(), "RAWReader");
  ASSERT_TRUE(reader.good());
  ASSERT_EQ(reader.frame(), SIZE_MAX);
  ASSERT_EQ(reader.depthImageRes(), Eigen::Vector2i(width_, height_));
  ASSERT_EQ(reader.RGBAImageRes(),  Eigen::Vector2i(width_, height_));

  // Read the dataset twice to test restart().
  for (size_t i = 0; i < 2; ++i) {
    // Read in all the frames and poses.
    se::Image<float> depth_image (1, 1);
    se::Image<uint32_t> rgba_image (1, 1);
    Eigen::Matrix4f T_WB;
    size_t frame = 0;
    while (reader.nextData(depth_image, rgba_image, T_WB) == se::ReaderStatus::ok) {
      // Test for correct state.
      ASSERT_TRUE(reader.good());
      ASSERT_EQ(reader.frame(), frame);

      // Test for correct image sizes.
      ASSERT_EQ(static_cast<size_t>(depth_image.width()), width_);
      ASSERT_EQ(static_cast<size_t>(depth_image.height()), height_);
      ASSERT_EQ(static_cast<size_t>(rgba_image.width()), width_);
      ASSERT_EQ(static_cast<size_t>(rgba_image.height()), height_);

      // Test for correct image data.
      for (size_t pixel = 0; pixel < depth_image.size(); ++pixel) {
        ASSERT_FLOAT_EQ(depth_image[pixel], frame + pixel);
        ASSERT_EQ(rgba_image[pixel], 0xFF000000 + frame + 10 * pixel);
      }

      // Test for correct pose.
      Eigen::Matrix4f T_WB_desired = Eigen::Matrix4f::Identity();
      T_WB_desired(0, 3) = frame;
      T_WB_desired(1, 3) = frame + 1;
      T_WB_desired(2, 3) = frame + 2;
      ASSERT_TRUE(T_WB.isApprox(T_WB_desired));

      frame++;
    }

    // Test for correct state.
    ASSERT_FALSE(reader.good());
    ASSERT_EQ(reader.frame(), num_frames_);

    // Restart and test for correct state.
    reader.restart();
    ASSERT_TRUE(reader.good());
    ASSERT_EQ(reader.frame(), SIZE_MAX);
  }
}

