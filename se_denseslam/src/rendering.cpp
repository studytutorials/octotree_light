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

#include "se/image_utils.hpp"



void renderRGBAKernel(uint32_t*                  output_RGBA_image_data,
                      const Eigen::Vector2i&     output_RGBA_image_res,
                      const se::Image<uint32_t>& input_RGBA_image) {

  TICKD("renderRGBAKernel");
  memcpy(output_RGBA_image_data, input_RGBA_image.data(),
         output_RGBA_image_res.prod() * sizeof(uint32_t));
  TOCK("renderRGBAKernel");
}



void renderDepthKernel(uint32_t*              depth_RGBA_image_data,
                       float*                 depth_image_data,
                       const Eigen::Vector2i& depth_RGBA_image_res,
                       const float            near_plane,
                       const float            far_plane) {

  TICKD("renderDepthKernel");
  se::depth_to_rgba(depth_RGBA_image_data, depth_image_data, depth_RGBA_image_res,
      near_plane, far_plane);
  TOCK("renderDepthKernel");
}



void renderTrackKernel(uint32_t*              tracking_RGBA_image_data,
                       const TrackData*       tracking_result_data,
                       const Eigen::Vector2i& tracking_RGBA_image_res) {

  TICKD("renderTrackKernel");

#pragma omp parallel for
  for (int y = 0; y < tracking_RGBA_image_res.y(); y++)
    for (int x = 0; x < tracking_RGBA_image_res.x(); x++) {
      const int pixel_idx = x + tracking_RGBA_image_res.x() * y;
      switch (tracking_result_data[pixel_idx].result) {
        case 1:
          // Gray
          tracking_RGBA_image_data[pixel_idx] = 0xFF808080;
          break;
        case -1:
          // Black
          tracking_RGBA_image_data[pixel_idx] = 0xFF000000;
          break;
        case -2:
          // Red
          tracking_RGBA_image_data[pixel_idx] = 0xFF0000FF;
          break;
        case -3:
          // Green
          tracking_RGBA_image_data[pixel_idx] = 0xFF00FF00;
          break;
        case -4:
          // Blue
          tracking_RGBA_image_data[pixel_idx] = 0xFFFF0000;
          break;
        case -5:
          // Yellow
          tracking_RGBA_image_data[pixel_idx] = 0xFF00FFFF;
          break;
        default:
          // Orange
          tracking_RGBA_image_data[pixel_idx] = 0xFF8080FF;
          break;
      }
    }
  TOCK("renderTrackKernel");
}



inline void printNormals(const se::Image<Eigen::Vector3f>& normals,
                         const char*                       filename) {

  unsigned char* normal_RGBA_image_data = new unsigned char [normals.width() * normals.height() * 4];
  for (int y = 0; y < normals.height(); ++y) {
    for (int x = 0; x < normals.width(); ++x){
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

