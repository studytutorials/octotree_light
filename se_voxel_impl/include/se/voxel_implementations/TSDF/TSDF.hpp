/*
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
#ifndef __TSDF_HPP
#define __TSDF_HPP

#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/continuous/volume_template.hpp"
#include "se/sensor_implementation.hpp"



/**
 * Kinect Fusion Truncated Signed Distance Function voxel implementation.
 */
struct TSDF {

  /**
   * The voxel type used as the template parameter for se::Octree.
   */
  struct VoxelType {
    /**
     * The struct stored in each se::Octree voxel.
     */
    struct VoxelData {
      float x; /**< The value of the TSDF. */
      float y; /**< The number of measurements integrated in the voxel. */
    };

    static inline VoxelData invalid()     { return {1.f, -1.f}; }
    static inline VoxelData initData() { return {1.f,  0.f}; }

    template <typename T>
    using MemoryPoolType = se::PagedMemoryPool<T>;
    template <typename ElemT>
    using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
  };



  /**
   * The normals must be inverted when rendering a TSDF map.
   */
  static constexpr bool invert_normals = true;

  /**
   * The maximum value of the weight factor TSDF::VoxelType::VoxelData::y.
   */
  static constexpr float max_weight = 100.f;



  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  static size_t buildAllocationList(
      se::Octree<TSDF::VoxelType>& map,
      const se::Image<float>&      depth_image,
      const Eigen::Matrix4f&       T_MC,
      const SensorImpl&            sensor,
      se::key_t*                   allocation_list,
      size_t                       reserved);



  /**
   * Integrate a depth image into the map.
   */
  static void integrate(se::Octree<TSDF::VoxelType>& map,
                        const se::Image<float>&      depth_image,
                        const Eigen::Matrix4f&       T_CM,
                        const SensorImpl&            sensor,
                        const unsigned               frame);



  /**
   * Cast a ray and return the point where the surface was hit.
   */
  static Eigen::Vector4f raycast(
      const VolumeTemplate<TSDF, se::Octree>& volume,
      const Eigen::Vector3f&                  ray_origin_M,
      const Eigen::Vector3f&                  ray_dir_M,
      const float                             near_plane,
      const float                             far_plane,
      const float                             mu,
      const float                             step,
      const float                             large_step);
};

#endif

