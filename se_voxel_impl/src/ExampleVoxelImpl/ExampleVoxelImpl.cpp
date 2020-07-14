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
#include "se/str_utils.hpp"



// Initialize static data members.
constexpr bool invert_normals = false;

void ExampleVoxelImpl::configure() {
  // set to default values
}

void ExampleVoxelImpl::configure(YAML::Node yaml_config) {
  configure()
  if (yaml_config.IsNull()) return;

  // set yaml value if parameter key is available
};

std::string ExampleVoxelImpl::printConfig() {
  std::stringstream out;
  out << str_utils::header_to_pretty_str("VOXEL IMPL") << "\n";
  out << str_utils::bool_to_pretty_str(ExampleVoxelImpl::invert_normals, "Invert normals") << "\n";
  return out.str();
}

// Implement static member functions.
size_t ExampleVoxelImpl::buildAllocationList(OctreeType&             map,
                                             const se::Image<float>& depth_image,
                                             const Eigen::Matrix4f&  T_MC,
                                             const SensorImpl&       sensor,
                                             se::key_t*              allocation_list,
                                             size_t                  reserved) {

  return 0;
}



void ExampleVoxelImpl::integrate(OctreeType&             map,
                                 const se::Image<float>& depth_image,
                                 const Eigen::Matrix4f&  T_CM,
                                 const SensorImpl&       sensor,
                                 const unsigned          frame) {
}



Eigen::Vector4f ExampleVoxelImpl::raycast(const OctreeType&      map,
                                          const Eigen::Vector3f& ray_origin_M,
                                          const Eigen::Vector3f& ray_dir_M,
                                          const float            t_near,
                                          const float            t_far) {
  return Eigen::Vector4f::Zero();
}

void ExampleVoxelImpl::dumpMesh(OctreeType&                map,
                                std::vector<se::Triangle>& mesh) {
}

