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

#include "se/voxel_implementations/ExampleVoxelImpl/ExampleVoxelImpl.hpp"



// Initialize static data members.
constexpr bool invert_normals = false;



// Implement static member functions.
size_t ExampleVoxelImpl::buildAllocationList(
    se::key_t*                               allocation_list,
    size_t                                   reserved,
    se::Octree<ExampleVoxelImpl::VoxelType>& map_index,
    const Eigen::Matrix4f&                   T_wc,
    const Eigen::Matrix4f&                   K,
    const float*                             depth_map,
    const Eigen::Vector2i&                   image_size,
    const float                              mu) {

  return 0;
}



void ExampleVoxelImpl::integrate(
    se::Octree<ExampleVoxelImpl::VoxelType>& map,
    const Sophus::SE3f&                      T_cw,
    const Eigen::Matrix4f&                   K,
    const se::Image<float>&                  depth,
    const float                              mu,
    const unsigned                           frame) {
}



Eigen::Vector4f ExampleVoxelImpl::raycast(
    const VolumeTemplate<ExampleVoxelImpl, se::Octree>& volume,
    const Eigen::Vector3f&                              origin,
    const Eigen::Vector3f&                              direction,
    const float                                         tnear,
    const float                                         tfar,
    const float                                         mu,
    const float                                         step,
    const float                                         large_step) {

  return Eigen::Vector4f::Zero();
}

