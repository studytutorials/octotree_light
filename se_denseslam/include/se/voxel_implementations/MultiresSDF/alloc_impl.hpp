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

#ifndef __SDF_ALLOC_IMPL_HPP
#define __SDF_ALLOC_IMPL_HPP

#include <se/utils/math_utils.h>
#include <se/node.hpp>
#include <se/octree.hpp>
#include <se/utils/morton_utils.hpp>



/*
 * \brief Given a depth map and camera matrix it computes the list of
 * voxels intersected but not allocated by the rays around the measurement m in
 * a region comprised between m +/- band.
 * \param allocation_list output list of keys corresponding to voxel blocks to
 * be allocated
 * \param reserved allocated size of allocation_list
 * \param map_index indexing structure used to index voxel blocks
 * \param T_wc camera extrinsics matrix
 * \param K camera intrinsics matrix
 * \param depth_map input depth map
 * \param image_size dimensions of depth_map
 * \param size discrete extent of the map, in number of voxels
 * \param voxelSize spacing between two consegutive voxels, in metric space
 * \param band maximum extent of the allocating region, per ray
 */
template <typename HashType>
size_t buildAllocationList(HashType*                allocation_list,
                           size_t                   reserved,
                           se::Octree<MultiresSDF>& map_index,
                           const Eigen::Matrix4f&   T_wc,
                           const Eigen::Matrix4f&   K,
                           const float*             depth_map,
                           const Eigen::Vector2i&   image_size,
                           const unsigned int       volume_size,
                           const float              volume_extent,
                           const float              mu) {

  const float voxel_size =  volume_extent / volume_size;
  const float band = 2.f * mu;
  const float inverse_voxel_size = 1.f / voxel_size;
  const unsigned block_scale = log2(size) - se::math::log2_const(se::VoxelBlock<MultiresSDF>::side);

  Eigen::Matrix4f invK = K.inverse();
  const Eigen::Matrix4f kPose = T_wc * invK;


#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
#else
  unsigned int voxelCount;
#endif

  const Eigen::Vector3f camera = T_wc.topRightCorner<3, 1>();
  const int numSteps = ceil(band*inverse_voxel_size);
  voxelCount = 0;
#pragma omp parallel for
  for (int y = 0; y < image_size.y(); ++y) {
    for (int x = 0; x < image_size.x(); ++x) {
      if(depth_map[x + y*image_size.x()] == 0)
        continue;
      const float depth = depth_map[x + y*image_size.x()];
      Eigen::Vector3f worldVertex = (kPose * Eigen::Vector3f((x + 0.5f) * depth,
            (y + 0.5f) * depth, depth).homogeneous()).head<3>();

      Eigen::Vector3f direction = (camera - worldVertex).normalized();
      const Eigen::Vector3f origin = worldVertex - (band * 0.5f) * direction;
      const Eigen::Vector3f step = (direction*band)/numSteps;

      Eigen::Vector3i voxel;
      Eigen::Vector3f voxelPos = origin;
      for (int i = 0; i < numSteps; i++) {
        Eigen::Vector3f voxelScaled = (voxelPos * inverse_voxel_size).array().floor();
        if ((voxelScaled.x() < size)
            && (voxelScaled.y() < size)
            && (voxelScaled.z() < size)
            && (voxelScaled.x() >= 0)
            && (voxelScaled.y() >= 0)
            && (voxelScaled.z() >= 0)) {
          voxel = voxelScaled.cast<int>();
          se::VoxelBlock<MultiresSDF> * n = map_index.fetch(voxel.x(),
              voxel.y(), voxel.z());
          if (!n) {
            HashType k = map_index.hash(voxel.x(), voxel.y(), voxel.z(),
                block_scale);
            unsigned int idx = voxelCount++;
            if(idx < reserved) {
              allocation_list[idx] = k;
            } else {
              break;
            }
          } else {
            n->active(true);
          }
        }
        voxelPos +=step;
      }
    }
  }
  const size_t written = voxelCount;
  return written >= reserved ? reserved : written;
}

#endif
