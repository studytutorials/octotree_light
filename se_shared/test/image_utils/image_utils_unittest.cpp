/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include <Eigen/Dense>

#include <se/image_utils.hpp>



class RGBAConversion : public ::testing::Test {
  protected:
    virtual void SetUp() {
    }

    static constexpr size_t num_values_ = 4;
    // Individual channels.
    const uint8_t  array_r_[num_values_] = {0x00, 0x01, 0xFB, 0xFF};
    const uint8_t  array_g_[num_values_] = {0x00, 0x02, 0xFC, 0xFF};
    const uint8_t  array_b_[num_values_] = {0x00, 0x03, 0xFD, 0xFF};
    const uint8_t  array_a_[num_values_] = {0x00, 0x04, 0xFE, 0xFF};
    const Eigen::Vector4i array_4i_[num_values_]
        = {{array_r_[0], array_g_[0], array_b_[0], array_a_[0]},
           {array_r_[1], array_g_[1], array_b_[1], array_a_[1]},
           {array_r_[2], array_g_[2], array_b_[2], array_a_[2]},
           {array_r_[3], array_g_[3], array_b_[3], array_a_[3]}};
    const Eigen::Vector4f array_4f_[num_values_]
        = {array_4i_[0].cast<float>() / 255.f,
           array_4i_[1].cast<float>() / 255.f,
           array_4i_[2].cast<float>() / 255.f,
           array_4i_[3].cast<float>() / 255.f};
    // Combined channels.
    const uint32_t array_rgba_[num_values_]
        = {0x00000000, 0x04030201, 0xFEFDFCFB, 0xFFFFFFFF};
};



TEST_F(RGBAConversion, Pack) {
  for (size_t i = 0; i < num_values_; ++i) {
    const uint32_t rgba
        = to_rgba(array_r_[i], array_g_[i], array_b_[i], array_a_[i]);
    EXPECT_EQ(rgba, array_rgba_[i]);

    const uint32_t rgba_4i = to_rgba(array_4i_[i]);
    EXPECT_EQ(rgba_4i, array_rgba_[i]);

    const uint32_t rgba_4f = to_rgba(array_4f_[i]);
    EXPECT_EQ(rgba_4f, array_rgba_[i]);
  }
}



TEST_F(RGBAConversion, Unpack) {
  for (size_t i = 0; i < num_values_; ++i) {
    const uint8_t r = r_from_rgba(array_rgba_[i]);
    const uint8_t g = g_from_rgba(array_rgba_[i]);
    const uint8_t b = b_from_rgba(array_rgba_[i]);
    const uint8_t a = a_from_rgba(array_rgba_[i]);
    EXPECT_EQ(r, array_r_[i]);
    EXPECT_EQ(g, array_g_[i]);
    EXPECT_EQ(b, array_b_[i]);
    EXPECT_EQ(a, array_a_[i]);
  }
}





TEST(RGBABlending, Blend) {
  const size_t num_values = 3;
  const uint32_t colors_a[num_values] = {0xFF0080C0, 0xFF0080C0, 0xFF0080C0};
  const uint32_t colors_b[num_values] = {0x00FF8020, 0x00FF8020, 0x00FF8020};
  const uint32_t colors_c[num_values] = {0x00FF8020, 0x80808070, 0xFF0080C0};
  const float factor[num_values]    = {0.f, 0.5f, 1.f};

  for (size_t i = 0; i < num_values; ++i) {
    const uint32_t blended = blend(colors_a[i], colors_b[i], factor[i]);
    EXPECT_EQ(blended, colors_c[i]);
  }
}





class DepthSaveLoad : public ::testing::Test {
  protected:
    virtual void SetUp() {
      depth_ = new uint16_t[depth_size_pixels_]();

      // Initialize the test image with a patter.
      for (size_t w = 0; w < depth_width_; ++w) {
        for (size_t h = 0; h < depth_height_; ++h) {
          if (w > h) {
            depth_[w + depth_width_ * h] = UINT16_MAX / 4;
          } else if (w == h) {
            depth_[w + depth_width_ * h] = UINT16_MAX / 2;
          } else {
            depth_[w + depth_width_ * h] = UINT16_MAX;
          }
        }
      }
    }

    uint16_t* depth_;
    const size_t depth_width_  = 64;
    const size_t depth_height_ = 64;
    const size_t depth_size_pixels_ = depth_width_ * depth_height_;
    const size_t depth_size_bytes_ = sizeof(uint16_t) * depth_size_pixels_;
    const Eigen::Vector2i depth_size_
      = Eigen::Vector2i(depth_width_, depth_height_);
};



TEST_F(DepthSaveLoad, SaveThenLoadPNG) {
  // Save the image.
  const int save_ok = save_depth_png(depth_, depth_size_, "/tmp/depth.png");
  EXPECT_EQ(save_ok, 0);

  // Load the image.
  uint16_t* depth_in;
  Eigen::Vector2i depth_in_size (0, 0);
  const int load_ok = load_depth_png(&depth_in, depth_in_size, "/tmp/depth.png");
  EXPECT_EQ(load_ok, 0);

  // Compare the loaded image with the saved one.
  EXPECT_EQ(depth_in_size.x(), depth_width_);
  EXPECT_EQ(depth_in_size.y(), depth_height_);
  EXPECT_EQ(memcmp(depth_in, depth_, depth_size_bytes_), 0);

  free(depth_in);
}



TEST_F(DepthSaveLoad, SaveThenLoadPGM) {
  // Save the image.
  const int save_ok = save_depth_pgm(depth_, depth_size_, "/tmp/depth.pgm");
  EXPECT_EQ(save_ok, 0);

  // Load the image.
  uint16_t* depth_in;
  Eigen::Vector2i depth_in_size (0, 0);
  const int load_ok = load_depth_pgm(&depth_in, depth_in_size, "/tmp/depth.pgm");
  EXPECT_EQ(load_ok, 0);

  // Compare the loaded image with the saved one.
  EXPECT_EQ(depth_in_size.x(), depth_width_);
  EXPECT_EQ(depth_in_size.y(), depth_height_);
  EXPECT_EQ(memcmp(depth_in, depth_, depth_size_bytes_), 0);

  free(depth_in);
}

