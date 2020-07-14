/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


 Copyright 2016 Emanuele Vespa, Imperial College London

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this
 list of conditions and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions and the following disclaimer in the documentation
 and/or other materials provided with the distribution.

 3. Neither the name of the copyright holder nor the names of its contributors
 may be used to endorse or promote products derived from this software without
 specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "se/preprocessing.hpp"

#include <cassert>

#include "se/image_utils.hpp"



void bilateralFilterKernel(se::Image<float>&         output_image,
                           const se::Image<float>&   input_image,
                           const std::vector<float>& gaussian,
                           const float               e_d,
                           const int                 radius) {

  if ((input_image.width() != output_image.width()) ||
       input_image.height() != output_image.height()) {
    std::cerr << "input/output image sizes differ." << std::endl;
    exit(1);
  }

  TICK()
  const int width = input_image.width();
  const int height = input_image.height();
  const float e_d_squared_2 = e_d * e_d * 2.f;
#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const unsigned int pixel_idx = x + y * width;
      if (input_image[pixel_idx] == 0) {
        output_image[pixel_idx] = 0;
        continue;
      }

      float factor_count = 0.0f;
      float filter_value_sum = 0.0f;

      const float centre_value = input_image[pixel_idx];

      for (int i = -radius; i <= radius; ++i) {
        for (int j = -radius; j <= radius; ++j) {
          const Eigen::Vector2i pixel_tmp = Eigen::Vector2i(
              se::math::clamp(x + i, 0, width  - 1),
              se::math::clamp(y + j, 0, height - 1));
          const float pixel_value_tmp = input_image[pixel_tmp.x() + pixel_tmp.y() * width];
          if (pixel_value_tmp > 0.f) {
            const float mod = se::math::sq(pixel_value_tmp - centre_value);
            const float factor = gaussian[i + radius]
                * gaussian[j + radius] * expf(-mod / e_d_squared_2);
            filter_value_sum += factor * pixel_value_tmp;
            factor_count += factor;
          }
        }
      }
      output_image[pixel_idx] = filter_value_sum / factor_count;
    }
  }
  TOCK("bilateralFilterKernel", width * height);
}



void depthToPointCloudKernel(se::Image<Eigen::Vector3f>& point_cloud_C,
                             const se::Image<float>&     depth_image,
                             const SensorImpl&           sensor) {

  TICK();
#pragma omp parallel for
  for (int y = 0; y < depth_image.height(); y++) {
    for (int x = 0; x < depth_image.width(); x++) {
      const Eigen::Vector2i pixel(x, y);
      if (depth_image(pixel.x(), pixel.y()) > 0) {
        const Eigen::Vector2f pixel_f = pixel.cast<float>();
        Eigen::Vector3f ray_dir_C;
        sensor.model.backProject(pixel_f, &ray_dir_C);
        point_cloud_C[pixel.x() + pixel.y() * depth_image.width()] = depth_image(pixel.x(), pixel.y()) * ray_dir_C;
      } else {
        point_cloud_C[pixel.x() + pixel.y() * depth_image.width()] = Eigen::Vector3f::Zero();
      }
    }
  }
  TOCK("depthToPointCloudKernel", depth_image.width() * depth_image.height());
}



void pointCloudToDepthKernel(se::Image<float>&            depth_image,
                        const se::Image<Eigen::Vector3f>& point_cloud_X,
                        const Eigen::Matrix4f&            T_CX) {

  TICK();
#pragma omp parallel for
  for (int y = 0; y < depth_image.height(); y++) {
    for (int x = 0; x < depth_image.width(); x++) {
      depth_image(x, y) = (T_CX * point_cloud_X(x, y).homogeneous()).z();
    }
  }
  TOCK("pointCloudToDepthKernel", depth_image.width() * depth_image.height());
}



// Explicit template instantiation
template void pointCloudToNormalKernel<true>(se::Image<Eigen::Vector3f>&       normals,
                                             const se::Image<Eigen::Vector3f>& point_cloud);
template void pointCloudToNormalKernel<false>(se::Image<Eigen::Vector3f>&       normals,
                                              const se::Image<Eigen::Vector3f>& point_cloud);

template <bool NegY>
void pointCloudToNormalKernel(se::Image<Eigen::Vector3f>&       normals,
                              const se::Image<Eigen::Vector3f>& point_cloud) {

  TICK();
  const int width = point_cloud.width();
  const int height = point_cloud.height();
#pragma omp parallel for
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      const Eigen::Vector3f point = point_cloud[x + width * y];
      if (point.z() == 0.f) {
        normals[x + y * width].x() = INVALID;
        continue;
      }

      const Eigen::Vector2i p_left
          = Eigen::Vector2i(std::max(int(x) - 1, 0), y);
      const Eigen::Vector2i p_right
          = Eigen::Vector2i(std::min(x + 1, (int) width - 1), y);

      // Swapped to match the left-handed coordinate system of ICL-NUIM
      Eigen::Vector2i p_up, p_down;
      if (NegY) {
        p_up   = Eigen::Vector2i(x, std::max(int(y) - 1, 0));
        p_down = Eigen::Vector2i(x, std::min(y + 1, ((int) height) - 1));
      } else {
        p_down = Eigen::Vector2i(x, std::max(int(y) - 1, 0));
        p_up   = Eigen::Vector2i(x, std::min(y + 1, ((int) height) - 1));
      }

      const Eigen::Vector3f left  = point_cloud[p_left.x()  + width * p_left.y()];
      const Eigen::Vector3f right = point_cloud[p_right.x() + width * p_right.y()];
      const Eigen::Vector3f up    = point_cloud[p_up.x()    + width * p_up.y()];
      const Eigen::Vector3f down  = point_cloud[p_down.x()  + width * p_down.y()];

      if (left.z() == 0 || right.z() == 0 || up.z() == 0 || down.z() == 0) {
        normals[x + y * width].x() = INVALID;
        continue;
      }
      const Eigen::Vector3f dv_x = right - left;
      const Eigen::Vector3f dv_y = up - down;
      normals[x + y * width] =  dv_x.cross(dv_y).normalized();
    }
  }
  TOCK("pointCloudToNormalKernel", width * height);
}



void mm2metersKernel(se::Image<float>&      output_depth_image,
                     const float*           input_depth_image_data,
                     const Eigen::Vector2i& input_depth_image_res) {
  TICK();
  // Check for unsupported conditions
  if ((input_depth_image_res.x() < output_depth_image.width()) ||
       input_depth_image_res.y() < output_depth_image.height()) {
    std::cerr << "Invalid ratio." << std::endl;
    exit(1);
  }
  if ((input_depth_image_res.x() % output_depth_image.width() != 0) ||
      (input_depth_image_res.y() % output_depth_image.height() != 0)) {
    std::cerr << "Invalid ratio." << std::endl;
    exit(1);
  }
  if ((input_depth_image_res.x() / output_depth_image.width() !=
       input_depth_image_res.y() / output_depth_image.height())) {
    std::cerr << "Invalid ratio." << std::endl;
    exit(1);
  }

  const int ratio = input_depth_image_res.x() / output_depth_image.width();
#pragma omp parallel for
  for (int y_out = 0; y_out < output_depth_image.height(); y_out++) {
    for (int x_out = 0; x_out < output_depth_image.width(); x_out++) {
      size_t valid_count = 0;
      float pixel_value_sum = 0;
      for (int b = 0; b < ratio; b++) {
        for (int a = 0; a < ratio; a++) {
          const int y_in = y_out * ratio + b;
          const int x_in = x_out * ratio + a;
          const float depth_value = input_depth_image_data[x_in + input_depth_image_res.x() * y_in];
          if ((depth_value == 0) || std::isnan(depth_value))
            continue;
          pixel_value_sum += depth_value;
          valid_count++;
        }
      }
      output_depth_image(x_out, y_out)
          = (valid_count > 0) ? pixel_value_sum / valid_count : 0;
    }
  }
  TOCK("mm2metersKernel", output_depth_image.width() * output_depth_image.height());
}



void halfSampleRobustImageKernel(se::Image<float>&       output_image,
                                 const se::Image<float>& input_image,
                                 const float             e_d,
                                 const int               radius) {

  if ((input_image.width() / output_image.width() != 2) ||
      (input_image.height() / output_image.height() != 2)) {
    std::cerr << "Invalid ratio." << std::endl;
    exit(1);
  }

  TICK();
#pragma omp parallel for
  for (int y = 0; y < output_image.height(); y++) {
    for (int x = 0; x < output_image.width(); x++) {
      const Eigen::Vector2i out_pixel = Eigen::Vector2i(x, y);
      const Eigen::Vector2i in_pixel = 2 * out_pixel;

      float pixel_count = 0.0f;
      float pixel_value_sum = 0.0f;
      const float in_pixel_value = input_image[in_pixel.x() + in_pixel.y() * input_image.width()];
      for (int i = -radius + 1; i <= radius; ++i) {
        for (int j = -radius + 1; j <= radius; ++j) {
          Eigen::Vector2i in_pixel_tmp = in_pixel + Eigen::Vector2i(j, i);
          se::math::clamp(in_pixel_tmp,
              Eigen::Vector2i::Zero(),
              Eigen::Vector2i(2 * output_image.width() - 1, 2 * output_image.height() - 1));
          const float in_pixel_value_tmp = input_image[in_pixel_tmp.x() + in_pixel_tmp.y() * input_image.width()];
          if (fabsf(in_pixel_value_tmp - in_pixel_value) < e_d) {
            pixel_count += 1.0f;
            pixel_value_sum += in_pixel_value_tmp;
          }
        }
      }
      output_image[out_pixel.x() + out_pixel.y() * output_image.width()] = pixel_value_sum / pixel_count;
    }
  }
  TOCK("halfSampleRobustImageKernel", output_image.width() * output_image.height());
}



void downsampleImageKernel(const uint32_t*        input_RGBA_image_data,
                           const Eigen::Vector2i& input_RGBA_image_res,
                           se::Image<uint32_t>&   output_RGBA_image) {

  TICK();
  // Check for correct image sizes.
  assert((input_RGBA_image_res.x() >= output_RGBA_image.width())
      && "Error: input width must be greater than output width");
  assert((input_RGBA_image_res.y() >= output_RGBA_image.height())
      && "Error: input height must be greater than output height");
  assert((input_RGBA_image_res.x() % output_RGBA_image.width() == 0)
      && "Error: input width must be an integer multiple of output width");
  assert((input_RGBA_image_res.y() % output_RGBA_image.height() == 0)
      && "Error: input height must be an integer multiple of output height");
  assert((input_RGBA_image_res.x() / output_RGBA_image.width()
      == input_RGBA_image_res.y() / output_RGBA_image.height())
      && "Error: input and output width and height ratios must be the same");

  const int ratio = input_RGBA_image_res.x() / output_RGBA_image.width();
  // Iterate over each output pixel.
#pragma omp parallel for
  for (int y_out = 0; y_out < output_RGBA_image.height(); ++y_out) {
    for (int x_out = 0; x_out < output_RGBA_image.width(); ++x_out) {

      // Average the neighboring pixels by iterating over the nearby input
      // pixels.
      uint16_t r = 0, g = 0, b = 0;
      for (int yy = 0; yy < ratio; ++yy) {
        for (int xx = 0; xx < ratio; ++xx) {
          const int x_in = x_out * ratio + xx;
          const int y_in = y_out * ratio + yy;
          const uint32_t pixel_value
              = input_RGBA_image_data[x_in + input_RGBA_image_res.x() * y_in];
          r += se::r_from_rgba(pixel_value);
          g += se::g_from_rgba(pixel_value);
          b += se::b_from_rgba(pixel_value);
        }
      }
      r /= ratio * ratio;
      g /= ratio * ratio;
      b /= ratio * ratio;

      // Combine into a uint32_t by adding an alpha channel with 100% opacity.
      const uint32_t rgba = se::pack_rgba(r, g, b, 255);
      output_RGBA_image(x_out, y_out) = rgba;
    }
  }
  TOCK("downsampleImageKernel", output_RGBA_image.width() * output_RGBA_image.height());
}

