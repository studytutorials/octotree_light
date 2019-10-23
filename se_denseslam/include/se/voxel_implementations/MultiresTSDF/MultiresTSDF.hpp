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

#ifndef __MULTIRESTSDF_HPP
#define __MULTIRESTSDF_HPP

#include <se/octree.hpp>
#include <se/image/image.hpp>
#include <se/continuous/volume_template.hpp>



/** Kinect Fusion Truncated Signed Distance Function voxel implementation for
 * integration at multipe scales. */
struct MultiresTSDF {

  /**
   * The voxel type used as the template parameter for se::Octree.
   */
  typedef struct {
    /**
     * The struct stored in each se::Octree voxel.
     */
    typedef struct {
      float x; /**< The value of the TSDF. */
      float x_last;
      int   y;
      int   delta_y;
    } VoxelData;

    static inline VoxelData empty()     { return {1.f, 1.f, 0, 0}; }
    static inline VoxelData initValue() { return {1.f, 1.f, 0, 0}; }
  } VoxelType;



  /**
   * The normals must be inverted when rendering a TSDF map.
   */
  static constexpr bool invert_normals = true;

  /**
   * The maximum value of the weight factor
   * MultiresTSDF::VoxelType::VoxelData::y.
   */
  static constexpr int max_weight = 100;



  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  template <typename HashType>
  static size_t buildAllocationList(
      HashType*                            allocation_list,
      size_t                               reserved,
      se::Octree<MultiresTSDF::VoxelType>& map_index,
      const Eigen::Matrix4f&               T_wc,
      const Eigen::Matrix4f&               K,
      const float*                         depth_map,
      const Eigen::Vector2i&               image_size,
      const float                          mu);



  /**
   * Integrate a depth image into the map.
   */
  static inline void integrate(se::Octree<MultiresTSDF::VoxelType>& map,
                               const Sophus::SE3f&                  T_cw,
                               const Eigen::Matrix4f&               K,
                               const se::Image<float>&              depth,
                               const float                          mu,
                               const unsigned                       frame);



  /**
   * Cast a ray and return the point where the surface was hit.
   */
  static inline Eigen::Vector4f raycast(
      const VolumeTemplate<MultiresTSDF, se::Octree>& volume,
      const Eigen::Vector3f&                          origin,
      const Eigen::Vector3f&                          direction,
      const float                                     tnear,
      const float                                     tfar,
      const float                                     mu,
      const float                                     step,
      const float                                     large_step);
};



#include "se/voxel_implementations/MultiresTSDF/alloc_impl.hpp"
#include "se/voxel_implementations/MultiresTSDF/rendering_impl.hpp"
#include "se/voxel_implementations/MultiresTSDF/mapping_impl.hpp"

#endif

