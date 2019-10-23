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

#ifndef _SDF_MAPPING_IMPL_HPP
#define _SDF_MAPPING_IMPL_HPP

#include <se/octree.hpp>
#include <se/node.hpp>
#include <se/functors/projective_functor.hpp>
#include "SDF.hpp"



struct sdf_update {
  const float* depth;
  Eigen::Vector2i depth_size;
  float mu;



  sdf_update(const float*           depth,
             const Eigen::Vector2i& depth_size,
             float                  mu)
    : depth(depth), depth_size(depth_size), mu(mu) {};



  template <typename DataHandlerT>
  void operator()(DataHandlerT&          handler,
                  const Eigen::Vector3i&,
                  const Eigen::Vector3f& pos,
                  const Eigen::Vector2f& pixel) {

    const Eigen::Vector2i px = pixel.cast<int>();
    const float depthSample = depth[px.x() + depth_size.x()*px.y()];
    // Return on invalid depth measurement
    if (depthSample <=  0)
      return;

    // Update the TSDF
    const float diff = (depthSample - pos.z())
      * std::sqrt( 1 + se::math::sq(pos.x() / pos.z()) + se::math::sq(pos.y() / pos.z()));
    if (diff > -mu) {
      const float sdf = fminf(1.f, diff / mu);
      auto data = handler.get();
      data.x = se::math::clamp(
          (static_cast<float>(data.y) * data.x + sdf) / (static_cast<float>(data.y) + 1.f),
          -1.f,
          1.f);
      data.y = fminf(data.y + 1, SDF::max_weight);
      handler.set(data);
    }
  }
};



inline void SDF::integrate(se::Octree<SDF::VoxelType>& map,
                           const Sophus::SE3f&         T_cw,
                           const Eigen::Matrix4f&      K,
                           const se::Image<float>&     depth,
                           const float                 mu,
                           const unsigned) {

  const Eigen::Vector2i depth_size (depth.width(), depth.height());

  struct sdf_update funct(depth.data(), depth_size, mu);

  se::functor::projective_map(map, map._offset, T_cw, K, depth_size, funct);
}

#endif

