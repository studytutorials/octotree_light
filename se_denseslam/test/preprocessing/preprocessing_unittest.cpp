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

#include <cstdint>
#include <cstring>

#include <gtest/gtest.h>
#include <lodepng.h>

#include <se/preprocessing.hpp>



// Needed due to the way the TICK and TOCK macros used in preprocessing are
// defined.
PerfStats stats;



void test_downsample_depth_kernel(const float*           input_depth,
                                  const Eigen::Vector2i& input_res,
                                  const float*           desired_depth,
                                  const Eigen::Vector2i& desired_res) {
  se::Image<float> output_depth (desired_res.x(), desired_res.y());
  downsampleDepthKernel(input_depth, input_res, output_depth);
  for (size_t i = 0; i < output_depth.size(); ++i) {
    EXPECT_FLOAT_EQ(output_depth[i], desired_depth[i]);
  }
}



TEST(DownsampleImageKernel, UniformImageHalf) {
  // 4x4 white image.
  const uint8_t input_RGBA[4*4*4] = {
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
  };

  const uint8_t desired_output_RGBA[2*2*4] = {
    255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,
  };

  se::Image<uint32_t> output_RGBA (2, 2);

  downsampleImageKernel(reinterpret_cast<const uint32_t*>(input_RGBA), Eigen::Vector2i(4, 4), output_RGBA);

  ASSERT_EQ(memcmp(desired_output_RGBA, output_RGBA.data(), 2*2*4), 0);
}



TEST(DownsampleImageKernel, UniformImageQuarter) {
  // 4x4 white image.
  const uint8_t input_RGBA[4*4*4] = {
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
    255,255,255,255,   255,255,255,255,   255,255,255,255,   255,255,255,255,
  };

  const uint8_t desired_output_RGBA[1*1*4] = {
    255,255,255,255,
  };

  se::Image<uint32_t> output_RGBA (1, 1);

  downsampleImageKernel(reinterpret_cast<const uint32_t*>(input_RGBA), Eigen::Vector2i(4, 4), output_RGBA);

  ASSERT_EQ(memcmp(desired_output_RGBA, output_RGBA.data(), 1*1*4), 0);
}



TEST(DownsampleImageKernel, VariedImageHalf) {
  // 4x4 image.
  // Red  Red  Green Green
  // Red  Red  Green Green
  // Blue Blue Black Black
  // Blue Blue Black Black
  const uint8_t input_RGBA[4*4*4] = {
    255,  0,  0,255,   255,  0,  0,255,     0,255,  0,255,     0,255,  0,255,
    255,  0,  0,255,   255,  0,  0,255,     0,255,  0,255,     0,255,  0,255,
      0,  0,255,255,     0,  0,255,255,     0,  0,  0,255,     0,  0,  0,255,
      0,  0,255,255,     0,  0,255,255,     0,  0,  0,255,     0,  0,  0,255,
  };

  const uint8_t desired_output_RGBA[2*2*4] = {
    255,  0,  0,255,     0,255,  0,255,
      0,  0,255,255,     0,  0,  0,255,
  };

  se::Image<uint32_t> output_RGBA (2, 2);

  downsampleImageKernel(reinterpret_cast<const uint32_t*>(input_RGBA), Eigen::Vector2i(4, 4), output_RGBA);

  ASSERT_EQ(memcmp(desired_output_RGBA, output_RGBA.data(), 2*2*4), 0);
}



TEST(DownsampleImageKernel, VariedImageQuarter) {
  // 4x4 image.
  // Red  Red  Green Green
  // Red  Red  Green Green
  // Blue Blue Black Black
  // Blue Blue Black Black
  const uint8_t input_RGBA[4*4*4] = {
    255,  0,  0,255,   255,  0,  0,255,     0,255,  0,255,     0,255,  0,255,
    255,  0,  0,255,   255,  0,  0,255,     0,255,  0,255,     0,255,  0,255,
      0,  0,255,255,     0,  0,255,255,     0,  0,  0,255,     0,  0,  0,255,
      0,  0,255,255,     0,  0,255,255,     0,  0,  0,255,     0,  0,  0,255,
  };

  const uint8_t desired_output_RGBA[1*1*4] = {
     63, 63, 63,255,
  };

  se::Image<uint32_t> output_RGBA (1, 1);

  downsampleImageKernel(reinterpret_cast<const uint32_t*>(input_RGBA), Eigen::Vector2i(4, 4), output_RGBA);

  ASSERT_EQ(memcmp(desired_output_RGBA, output_RGBA.data(), 1*1*4), 0);
}



TEST(DownsampleDepthKernel, UniformImageHalf) {
  const Eigen::Vector2i input_res (4, 4);
  const float input_depth[4 * 4] = {
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
  };

  const float desired_depth[2 * 2] = {
    2.0f, 2.0f,
    2.0f, 2.0f,
  };

  test_downsample_depth_kernel(input_depth, Eigen::Vector2i(4, 4),
      desired_depth, Eigen::Vector2i(2, 2));
}



TEST(DownsampleDepthKernel, UniformImageQuarter) {
  const float input_depth[4 * 4] = {
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
    2.0f, 2.0f, 2.0f, 2.0f,
  };

  const float desired_depth[1 * 1] = {
    2.0f,
  };

  test_downsample_depth_kernel(input_depth, Eigen::Vector2i(4, 4),
      desired_depth, Eigen::Vector2i(1, 1));
}



TEST(DownsampleDepthKernel, VariedImageHalf) {
  const float input_depth[4 * 4] = {
    1.0f,   1.5f,     8.0f,     1.0f,
    3.0f,   0.0f,     2.4f,     6.2f,
    4.0f,   3.0f,     0.0f,     0.0f,
    2.0f,   3.0f,     0.0f,     0.0f,
  };

  const float desired_depth[2 * 2] = {
    1.5f, 6.2f,
    3.0f, 0.0f,
  };

  test_downsample_depth_kernel(input_depth, Eigen::Vector2i(4, 4),
      desired_depth, Eigen::Vector2i(2, 2));
}



TEST(DownsampleDepthKernel, VariedImageQuarter) {
  const float input_depth[4 * 4] = {
    1.0f,   1.0f,     2.0f,     2.0f,
    1.0f,   1.0f,     2.0f,     2.0f,
    3.0f,   3.0f,     0.0f,     0.0f,
    3.0f,   3.0f,     0.0f,     0.0f,
  };

  const float desired_depth[1 * 1] = {
     2.0f,
  };

  test_downsample_depth_kernel(input_depth, Eigen::Vector2i(4, 4),
      desired_depth, Eigen::Vector2i(1, 1));
}

