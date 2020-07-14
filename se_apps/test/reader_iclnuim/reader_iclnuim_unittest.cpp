/*
 * SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include "lodepng.h"

#include "reader_iclnuim.hpp"
#include "filesystem.hpp"
#include "se/image_utils.hpp"



class TestICLNUIMDataset : public ::testing::Test {
  protected:
    virtual void SetUp() {
      std::string tmpdir (stdfs::temp_directory_path());
      // Create a temporary dataset directory.
      std::string scene_dirname_ = tmpdir + "/test_scene";
      stdfs::create_directory(scene_dirname_);

      // Generate depth and RGB images.
      for (size_t frame = 0; frame < num_frames_; ++frame) {
        // Generate the filenames for the current frame.
        std::ostringstream basename;
        basename << "scene_00_" << std::setfill('0') << std::setw(4) << frame;
        const std::string basepath (scene_dirname_ + "/" + basename.str());
        const std::string depth_filename (basepath + ".depth");
        const std::string rgb_filename (basepath + ".png");

        // Generate the depth image. It is just a text file with
        // space-separated float values.
        std::ofstream fs (depth_filename, std::ios::out);
        fs << std::showpoint;
        for (size_t pixel = 0; pixel < width_ * height_; ++pixel) {
          fs << std::setprecision(10)
             << static_cast<float>(frame + pixel) << " ";
        }
        fs.close();

        // Generate a 16-bit RGB PNG image.
        std::unique_ptr<uint16_t[]> rgb_data (
            new uint16_t[3 * width_ * height_]);
        for (size_t pixel = 0; pixel < width_ * height_; ++pixel) {
          // Set the red, green and blue channels separately.
          rgb_data[3 * pixel + 0] = static_cast<uint8_t>(frame + pixel +  64);
          rgb_data[3 * pixel + 1] = static_cast<uint8_t>(frame + pixel + 128);
          rgb_data[3 * pixel + 2] = static_cast<uint8_t>(frame + pixel + 192);
        }
        lodepng_encode_file(rgb_filename.c_str(),
            reinterpret_cast<const unsigned char*>(rgb_data.get()),
            width_, height_, LCT_RGB, 16);
      }

      // Create a new groundtruth .txt file in the temporary directory.
      gt_filename_ = scene_dirname_ + "/test_scene.gt.freiburg";
      std::ofstream fs (gt_filename_, std::ios::out);
      // Generate the groundtruth .txt file.
      fs << std::showpoint;
      for (size_t frame = 0; frame < num_frames_; ++frame) {
        fs << "some garbage data "
           << frame << " " << frame + 1 << " " << frame + 2 << " "
           << 0.f << " " << 0.f << " " << 0.f << " " << 1.f << "\n";
      }
      fs.close();

      // Generate the reader config.
      config_ = {0, false, scene_dirname_, gt_filename_};
    }

    const size_t num_frames_   = 2;
    const uint32_t width_      = 640;
    const uint32_t height_     = 480;
    std::string scene_dirname_;
    std::string gt_filename_;
    se::ReaderConfig config_;
};



TEST_F(TestICLNUIMDataset, ReadDepthRGBAPose) {
  // Initialize the reader.
  se::ICLNUIMReader reader (config_);

  // Test for correct initial state.
  ASSERT_EQ(reader.name(), "ICLNUIMReader");
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
      for (int y = 0; y < depth_image.height(); ++y) {
        for (int x = 0; x < depth_image.width(); ++x) {
          const int pixel = x + y * depth_image.width();
          const float expected_depth
              = se::ICLNUIMReader::distanceToDepth(frame + pixel, x, y);
          ASSERT_FLOAT_EQ(depth_image(x, y), expected_depth);
          ASSERT_EQ(se::r_from_rgba(rgba_image[pixel]),
              static_cast<uint8_t>(frame + pixel +  64));
          ASSERT_EQ(se::g_from_rgba(rgba_image[pixel]),
              static_cast<uint8_t>(frame + pixel + 128));
          ASSERT_EQ(se::b_from_rgba(rgba_image[pixel]),
              static_cast<uint8_t>(frame + pixel + 192));
        }
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

