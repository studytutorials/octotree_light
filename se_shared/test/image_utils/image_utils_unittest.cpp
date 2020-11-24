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
#include <memory>

#include <Eigen/Dense>

#include <se/image_utils.hpp>



class RGBAPixelConversion : public ::testing::Test {
  protected:
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



TEST_F(RGBAPixelConversion, Pack) {
  for (size_t i = 0; i < num_values_; ++i) {
    const uint32_t rgba
        = se::pack_rgba(array_r_[i], array_g_[i], array_b_[i], array_a_[i]);
    EXPECT_EQ(rgba, array_rgba_[i]);

    const uint32_t rgba_4i = se::pack_rgba(array_4i_[i]);
    EXPECT_EQ(rgba_4i, array_rgba_[i]);

    const uint32_t rgba_4f = se::pack_rgba(array_4f_[i]);
    EXPECT_EQ(rgba_4f, array_rgba_[i]);
  }
}



TEST_F(RGBAPixelConversion, Unpack) {
  for (size_t i = 0; i < num_values_; ++i) {
    const uint8_t r = se::r_from_rgba(array_rgba_[i]);
    const uint8_t g = se::g_from_rgba(array_rgba_[i]);
    const uint8_t b = se::b_from_rgba(array_rgba_[i]);
    const uint8_t a = se::a_from_rgba(array_rgba_[i]);
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
    const uint32_t blended = se::blend(colors_a[i], colors_b[i], factor[i]);
    EXPECT_EQ(blended, colors_c[i]);
  }
}





class RGBAImageConversion : public ::testing::Test {
  protected:
    virtual void SetUp() {
      rgb_ = std::unique_ptr<uint8_t[]>(new uint8_t[3 * num_pixels_]());
      rgba_ = std::unique_ptr<uint32_t[]>(new uint32_t[num_pixels_]());

      // Initialize the test images with a pattern.
      for (size_t p = 0; p < num_pixels_; ++p) {
        const uint8_t r = 0;
        const uint8_t g = 0;
        const uint8_t b = 0;
        rgb_.get()[3*p + 0] = r;
        rgb_.get()[3*p + 1] = g;
        rgb_.get()[3*p + 2] = b;
        rgba_.get()[p] = se::pack_rgba(r, g, b, 0xFF);
      }
    }

    std::unique_ptr<uint8_t[]> rgb_;
    std::unique_ptr<uint32_t[]> rgba_;
    const size_t width_  = 64;
    const size_t height_ = 64;
    const size_t num_pixels_ = width_ * height_;
    const size_t rgb_size_bytes_ = 3 * sizeof(uint8_t) * num_pixels_;
    const size_t rgba_size_bytes_ = sizeof(uint32_t) * num_pixels_;
};



TEST_F(RGBAImageConversion, RGBToRGBA) {
  std::unique_ptr<uint32_t[]> rgba (new uint32_t[num_pixels_]());
  se::rgb_to_rgba(rgb_.get(), rgba.get(), num_pixels_);
  EXPECT_EQ(memcmp(rgba.get(), rgba_.get(), rgba_size_bytes_), 0);
}



TEST_F(RGBAImageConversion, RGBAToRGB) {
  std::unique_ptr<uint8_t[]> rgb (new uint8_t[3 * num_pixels_]());
  se::rgba_to_rgb(rgba_.get(), rgb.get(), num_pixels_);
  EXPECT_EQ(memcmp(rgb.get(), rgb_.get(), rgb_size_bytes_), 0);
}

