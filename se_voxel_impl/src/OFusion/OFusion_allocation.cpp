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
 *
 * */

#include "se/voxel_implementations/OFusion/OFusion.hpp"

#include "se/utils/math_utils.h"
#include "se/node.hpp"



/* Compute step size based on distance travelled along the ray */
inline float ofusion_compute_step_size(const float dist_travelled,
                                       const float band,
                                       const float voxel_dim) {

  float new_step_size;
  float half_band = band * 0.5f;
  if (dist_travelled < band) {
    new_step_size = voxel_dim;
  } else if (dist_travelled < band + half_band) {
    new_step_size = 10.f * voxel_dim;
  } else {
    new_step_size = 30.f * voxel_dim;
  }
  return new_step_size;
}



/* Compute octree level given a step size */
inline int ofusion_step_to_depth(const float step,
                                 const int   voxel_depth,
                                 const float voxel_dim) {

  return static_cast<int>(floorf(std::log2f(voxel_dim / step)) + voxel_depth);
}



size_t OFusion::buildAllocationList(OctreeType&             map,
                                    const se::Image<float>& depth_image,
                                    const Eigen::Matrix4f&  T_MC,
                                    const SensorImpl&       sensor,
                                    se::key_t*              allocation_list,
                                    size_t                  reserved) {

  const Eigen::Vector2i depth_image_res (depth_image.width(), depth_image.height());
  const float voxel_dim = map.dim() / map.size();
  const float inverse_voxel_dim = 1.f / voxel_dim;
  const int map_size = map.size();
  const int voxel_depth = map.voxelDepth();
  const int block_depth = map.blockDepth();

#ifdef _OPENMP
  std::atomic<unsigned int> voxel_count (0);
#else
  unsigned int voxel_count = 0;
#endif

  const Eigen::Vector3f t_MC = T_MC.topRightCorner<3, 1>();
#pragma omp parallel for
  for (int y = 0; y < depth_image_res.y(); ++y) {
    for (int x = 0; x < depth_image_res.x(); ++x) {
      const Eigen::Vector2i pixel(x, y);
      const float depth_value_orig = depth_image(pixel.x(), pixel.y());
      if (depth_value_orig < sensor.near_plane) {
        continue;
      }
      const float depth_value = (depth_value_orig <= sensor.far_plane) ? depth_value_orig : sensor.far_plane;

      int depth = voxel_depth;
      float step_size = voxel_dim;

      Eigen::Vector3f ray_dir_C;
      const Eigen::Vector2f pixel_f = pixel.cast<float>();
      sensor.model.backProject(pixel_f, &ray_dir_C);
      const Eigen::Vector3f surface_vertex_M = (T_MC * (depth_value * ray_dir_C).homogeneous()).head<3>();

      const Eigen::Vector3f reverse_ray_dir_M = (t_MC - surface_vertex_M).normalized();
      const float sigma = se::math::clamp(OFusion::k_sigma * se::math::sq(depth_value), OFusion::sigma_min, OFusion::sigma_max);
      const float band = 2 * sigma;
      const Eigen::Vector3f ray_origin_M = surface_vertex_M - (band * 0.5f) * reverse_ray_dir_M;
      const float dist = (t_MC - ray_origin_M).norm();
      Eigen::Vector3f step = reverse_ray_dir_M * step_size;

      Eigen::Vector3f ray_pos_M = ray_origin_M;
      float travelled = 0.f;
      for (; travelled < dist; travelled += step_size) {

        const Eigen::Vector3i voxel_coord
            = (ray_pos_M * inverse_voxel_dim).cast<int>();
        if (   (voxel_coord.x() < map_size)
            && (voxel_coord.y() < map_size)
            && (voxel_coord.z() < map_size)
            && (voxel_coord.x() >= 0)
            && (voxel_coord.y() >= 0)
            && (voxel_coord.z() >= 0)) {
          auto node_ptr = map.fetchNode(
              voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
          if (node_ptr == nullptr) {
            const se::key_t voxel_key = map.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(),
                std::min(depth, block_depth));
            const unsigned int idx = voxel_count++;
            if (idx < reserved) {
              allocation_list[idx] = voxel_key;
            }
          } else if (depth >= block_depth) {
            static_cast<VoxelBlockType*>(node_ptr)->active(true);
          }
        }

        step_size = ofusion_compute_step_size(travelled, band, voxel_dim);
        depth = ofusion_step_to_depth(step_size, voxel_depth, voxel_dim);

        step = reverse_ray_dir_M * step_size;
        ray_pos_M += step;
      }
    }
  }
  return (size_t) voxel_count >= reserved ? reserved : (size_t) voxel_count;
}

