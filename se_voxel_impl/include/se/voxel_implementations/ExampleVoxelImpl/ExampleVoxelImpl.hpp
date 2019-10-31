/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef __EXAMPLE_VOXEL_IMPL_HPP
#define __EXAMPLE_VOXEL_IMPL_HPP

#include <se/octree.hpp>
#include <se/continuous/volume_template.hpp>



/**
 * Minimal example of the structure of a potential voxel implementation. All
 * functions and data members are required. The signature of the functions
 * should not be changed. Additional static functions or data members may be
 * added freely.
 *
 * \note There is usually no reason to create an instance of this struct, it is
 * only meant to be passed as a template parameter to se::DenseSLAMSystem and
 * related functions/classes.
 */
struct ExampleVoxelImpl {

  /**
   * The voxel type used as the template parameter for se::Octree.
   *
   *
   * \note There is usually no reason to create an instance of this struct, it
   * is only meant to be passed as a template parameter to se::Octree and
   * related functions/classes.
   *
   * \warning The struct name must always be `VoxelType`.
   */
  struct VoxelType {
  /**
   * The declaration of the struct stored in each se::Octree voxel. It may
   * contain additional members if desired.
   *
   * \warning The struct name must always be `VoxelData`.
   */
    struct VoxelData {
      float  x; /**< The value stored in each voxel of the octree. */

      // Any other data stored in each voxel go here. Make sure to also update
      // empty() and initValue() to initialize all data members.
    };


    /**
     * Returns a value corresponding to invalid voxels.
     *
     * \warning The function signature must not be changed.
     */
    static inline VoxelData empty()     { return 0.f; }



    /**
     * Returns the value stored in newly created voxels.
     *
     * \warning The function signature must not be changed.
     */
    static inline VoxelData initValue() { return 1.f; }
  };



  /**
   * Set to true for TSDF maps, false for occupancy maps.
   *
   * \warning The name of this variable must always be `invert_normals`.
   */
  static constexpr bool invert_normals = false;

  // Any other constant parameters required for the implementation go here.



  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   *
   * \warning The function signature must not be changed.
   */
  static size_t buildAllocationList(
      se::key_t*                               allocation_list,
      size_t                                   reserved,
      se::Octree<ExampleVoxelImpl::VoxelType>& map_index,
      const Eigen::Matrix4f&                   T_wc,
      const Eigen::Matrix4f&                   K,
      const float*                             depth_map,
      const Eigen::Vector2i&                   image_size,
      const unsigned int                       volume_size,
      const float                              volume_extent,
      const float                              mu);



  /**
   * Integrate a depth image into the map.
   *
   * \warning The function signature must not be changed.
   */
  static inline void integrate(se::Octree<ExampleVoxelImpl::VoxelType>& map,
                               const Sophus::SE3f&                      T_cw,
                               const Eigen::Matrix4f&                   K,
                               const se::Image<float>&                  depth,
                               const float                              mu,
                               const unsigned                           frame);



  /**
   * Cast a ray and return the point where the surface was hit.
   *
   * \warning The function signature must not be changed.
   */
  static inline Eigen::Vector4f raycast(
      const VolumeTemplate<ExampleVoxelImpl, se::Octree>& volume,
      const Eigen::Vector3f&                              origin,
      const Eigen::Vector3f&                              direction,
      const float                                         tnear,
      const float                                         tfar,
      const float                                         mu,
      const float                                         step,
      const float                                         large_step);



  // Any other static functions required for the implementation go here.
};

#endif

