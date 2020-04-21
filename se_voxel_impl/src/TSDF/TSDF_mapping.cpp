/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * */

#include "se/voxel_implementations/TSDF/TSDF.hpp"

#include <algorithm>

#include "se/octree.hpp"
#include "se/node.hpp"
#include "se/projective_functor.hpp"
#include "se/image_utils.hpp"



struct tsdf_update {
  const se::Image<float>& depth_image;
  float mu;



  tsdf_update(const se::Image<float>& depth_image,
              float                   mu)
    : depth_image(depth_image), mu(mu) {};



  template <typename DataHandlerT>
  void operator()(DataHandlerT&          handler,
                  const Eigen::Vector3i&,
                  const Eigen::Vector3f& point_C,
                  const Eigen::Vector2f& pixel_f) {

    const Eigen::Vector2i pixel = round_pixel(pixel_f);
    const float depth_value = depth_image(pixel.x(), pixel.y());
    // Return on invalid depth measurement
    if (depth_value <= 0.f)
      return;

    // Update the TSDF
    const float point_dist = (depth_value - point_C.z())
      * std::sqrt(1 + se::math::sq(point_C.x() / point_C.z())
      + se::math::sq(point_C.y() / point_C.z()));
    if (point_dist > -mu) {
      const float tsdf = fminf(1.f, point_dist / mu);
      auto data = handler.get();
      data.x = (data.y * data.x + tsdf) / (data.y + 1.f);
      data.x = se::math::clamp(data.x, -1.f, 1.f);
      data.y = fminf(data.y + 1, TSDF::max_weight);
      handler.set(data);
    }
  }
};



void TSDF::integrate(se::Octree<TSDF::VoxelType>& map,
                     const se::Image<float>&      depth_image,
                     const Eigen::Matrix4f&       T_CM,
                     const SensorImpl&            sensor,
                     const unsigned) {

  const Eigen::Vector2i depth_image_res(depth_image.width(), depth_image.height());

  struct tsdf_update funct(depth_image, sensor.mu);

  se::functor::projective_octree(map, map.sample_offset_frac_, T_CM, sensor, depth_image_res, funct);
}

