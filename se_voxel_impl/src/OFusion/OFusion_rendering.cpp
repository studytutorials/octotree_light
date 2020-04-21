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

#include "se/voxel_implementations/OFusion/OFusion.hpp"

#include "se/utils/math_utils.h"
#include <type_traits>



Eigen::Vector4f OFusion::raycast(
    const VolumeTemplate<OFusion, se::Octree>& volume,
    const Eigen::Vector3f&                     ray_origin_M,
    const Eigen::Vector3f&                     ray_dir_M,
    const float                                near_plane,
    const float                                far_plane,
    const float                                mu,
    const float                                step,
    const float                                large_step) {

  auto select_node_occupancy = [](const auto& data){ return OFusion::VoxelType::initData().x; };
  auto select_voxel_occupancy = [](const auto& data){ return data.x; };
  if (near_plane < far_plane) {
    float t = near_plane;
    float step_size = step;
    float f_t = volume.interp(ray_origin_M + ray_dir_M * t, select_node_occupancy, select_voxel_occupancy).first;
    float f_tt = 0;
    int scale = 0;

    // if we are not already in it
    if (f_t <= OFusion::surface_boundary) {
      for (; t < far_plane; t += step_size) {
        const Eigen::Vector3f ray_pos_M = ray_origin_M + ray_dir_M * t;
        OFusion::VoxelType::VoxelData voxel_data = volume.get(ray_pos_M);
        if (voxel_data.x > -100.f && voxel_data.y > 0.f) {
          f_tt = volume.interp(ray_origin_M + ray_dir_M * t, select_node_occupancy, select_voxel_occupancy).first;
        }
        if (f_tt > OFusion::surface_boundary)
          break;
        f_t = f_tt;
      }
      if (f_tt > OFusion::surface_boundary) {
        // got it, calculate accurate intersection
        t = t - step_size * (f_tt - OFusion::surface_boundary) / (f_tt - f_t);
        Eigen::Vector4f res = (ray_origin_M + ray_dir_M * t).homogeneous();
        res.w() = scale;
        return res;
      }
    }
  }
  return Eigen::Vector4f::Zero();
}

