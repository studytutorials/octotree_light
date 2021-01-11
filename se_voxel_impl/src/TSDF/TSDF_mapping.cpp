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



struct TSDFUpdate {
  const SensorImpl& sensor_;



  TSDFUpdate(const SensorImpl& sensor) :
      sensor_(sensor) {};

  template <typename DataType,
      template <typename DataT> class VoxelBlockT>
  void reset(VoxelBlockT<DataType>* /* block */) {}

  template <typename DataType,
      template <typename DataT> class VoxelBlockT>
  void operator()(VoxelBlockT<DataType>* block, const bool is_visible) {
    block->active(is_visible);
  }

  template <typename DataHandlerT>
  void operator()(DataHandlerT&          handler,
                  const Eigen::Vector3f& point_C,
                  const float            depth_value) {

    // Update the TSDF
    const float m = sensor_.measurementFromPoint(point_C);
    const float sdf_value = (depth_value - m) / m * point_C.norm();
    if (sdf_value > -TSDF::mu) {
      const float tsdf_value = fminf(1.f, sdf_value / TSDF::mu);
      auto data = handler.get();
      data.x = (data.y * data.x + tsdf_value) / (data.y + 1.f);
      data.x = se::math::clamp(data.x, -1.f, 1.f);
      data.y = fminf(data.y + 1, TSDF::max_weight);
      handler.set(data);
    }
  }
};



void TSDF::integrate(OctreeType&             map,
                     const se::Image<float>& depth_image,
                     const Eigen::Matrix4f&  T_CM,
                     const SensorImpl&       sensor,
                     const unsigned, int) {

  struct TSDFUpdate funct(sensor);

  se::functor::projective_octree(map, map.sample_offset_frac_, T_CM, sensor, depth_image, funct);
}

