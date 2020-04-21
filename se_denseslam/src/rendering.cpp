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

#include "se/rendering.hpp"

#include <cstring>


void renderRGBAKernel(uint8_t*                   output_RGBA_image_data,
                      const Eigen::Vector2i&     output_RGBA_image_res,
                      const se::Image<uint32_t>& input_RGBA_image) {

  TICK();

  memcpy(output_RGBA_image_data, input_RGBA_image.data(),
         output_RGBA_image_res.x() * output_RGBA_image_res.y() * 4);

  TOCK("renderRGBAKernel", output_RGBA_image_res.x() * output_RGBA_image_res.y());
}



void renderDepthKernel(unsigned char*         depth_RGBA_image_data,
                       float*                 depth_image_data,
                       const Eigen::Vector2i& depth_RGBA_image_res,
                       const float            near_plane,
                       const float            far_plane) {

  TICK();

  const float range_scale = 1.f / (far_plane - near_plane);

#pragma omp parallel for
  for (int y = 0; y < depth_RGBA_image_res.y(); y++) {
    const int row_offset = y * depth_RGBA_image_res.x();
    for (int x = 0; x < depth_RGBA_image_res.x(); x++) {

      const unsigned int pixel_idx = row_offset + x;
      const unsigned int rgba_idx = pixel_idx * 4;

      if (depth_image_data[pixel_idx] < near_plane) {
        depth_RGBA_image_data[rgba_idx + 0] = 255;
        depth_RGBA_image_data[rgba_idx + 1] = 255;
        depth_RGBA_image_data[rgba_idx + 2] = 255;
        depth_RGBA_image_data[rgba_idx + 3] = 255;
      } else if (depth_image_data[pixel_idx] > far_plane) {
        depth_RGBA_image_data[rgba_idx + 0] = 0;
        depth_RGBA_image_data[rgba_idx + 1] = 0;
        depth_RGBA_image_data[rgba_idx + 2] = 0;
        depth_RGBA_image_data[rgba_idx + 3] = 255;
      } else {
        const float depth_value = (depth_image_data[pixel_idx] - near_plane) * range_scale;
        unsigned char rgba[4];
        gs2rgb(depth_value, rgba);
        depth_RGBA_image_data[rgba_idx + 0] = rgba[0];
        depth_RGBA_image_data[rgba_idx + 1] = rgba[1];
        depth_RGBA_image_data[rgba_idx + 2] = rgba[2];
        depth_RGBA_image_data[rgba_idx + 3] = rgba[3];
      }
    }
  }
  TOCK("renderDepthKernel", depth_RGBA_image_res.x() * depth_RGBA_image_res.y());
}



void renderTrackKernel(unsigned char*         tracking_RGBA_image_data,
                       const TrackData*       tracking_result_data,
                       const Eigen::Vector2i& tracking_RGBA_image_res) {

  TICK();

#pragma omp parallel for
  for (int y = 0; y < tracking_RGBA_image_res.y(); y++)
    for (int x = 0; x < tracking_RGBA_image_res.x(); x++) {
      const int pixel_idx = x + tracking_RGBA_image_res.x() * y;
      const int rgba_idx = pixel_idx * 4;
      switch (tracking_result_data[pixel_idx].result) {
        case 1:
          tracking_RGBA_image_data[rgba_idx + 0] = 128;
          tracking_RGBA_image_data[rgba_idx + 1] = 128;
          tracking_RGBA_image_data[rgba_idx + 2] = 128;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        case -1:
          tracking_RGBA_image_data[rgba_idx + 0] = 0;
          tracking_RGBA_image_data[rgba_idx + 1] = 0;
          tracking_RGBA_image_data[rgba_idx + 2] = 0;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        case -2:
          tracking_RGBA_image_data[rgba_idx + 0] = 255;
          tracking_RGBA_image_data[rgba_idx + 1] = 0;
          tracking_RGBA_image_data[rgba_idx + 2] = 0;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        case -3:
          tracking_RGBA_image_data[rgba_idx + 0] = 0;
          tracking_RGBA_image_data[rgba_idx + 1] = 255;
          tracking_RGBA_image_data[rgba_idx + 2] = 0;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        case -4:
          tracking_RGBA_image_data[rgba_idx + 0] = 0;
          tracking_RGBA_image_data[rgba_idx + 1] = 0;
          tracking_RGBA_image_data[rgba_idx + 2] = 255;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        case -5:
          tracking_RGBA_image_data[rgba_idx + 0] = 255;
          tracking_RGBA_image_data[rgba_idx + 1] = 255;
          tracking_RGBA_image_data[rgba_idx + 2] = 0;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
        default:
          tracking_RGBA_image_data[rgba_idx + 0] = 255;
          tracking_RGBA_image_data[rgba_idx + 1] = 128;
          tracking_RGBA_image_data[rgba_idx + 2] = 128;
          tracking_RGBA_image_data[rgba_idx + 3] = 255;
          break;
      }
    }
  TOCK("renderTrackKernel", tracking_RGBA_image_res.x() * tracking_RGBA_image_res.y());
}



inline void printNormals(const se::Image<Eigen::Vector3f>& normals,
                         const char*                       filename) {

  unsigned char* normal_RGBA_image_data = new unsigned char [normals.width() * normals.height() * 4];
  for (unsigned int y = 0; y < normals.height(); ++y) {
    for (unsigned int x = 0; x < normals.width(); ++x){
      const Eigen::Vector3f n = normals[x + normals.width() * y];
      normal_RGBA_image_data[4 * normals.width() * y + 4 * x + 0] = (n.x() / 2 + 0.5) * 255;
      normal_RGBA_image_data[4 * normals.width() * y + 4 * x + 1] = (n.y() / 2 + 0.5) * 255;
      normal_RGBA_image_data[4 * normals.width() * y + 4 * x + 2] = (n.z() / 2 + 0.5) * 255;
      normal_RGBA_image_data[4 * normals.width() * y + 4 * x + 3] = 255;
    }
  }
  lodepng_encode32_file(std::string(filename).append(".png").c_str(),
                        normal_RGBA_image_data , normals.width(), normals.height());
}

