/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


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

#include "se/DenseSLAMSystem.h"

#include <cstring>

#include "se/voxel_block_ray_iterator.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/io/meshing_io.hpp"
#include "se/io/octree_io.hpp"
#include "se/geometry/octree_collision.hpp"
#include "se/algorithms/balancing.hpp"
#include "se/functors/for_each.hpp"
#include "se/timings.h"
#include "se/perfstats.h"
#include "se/rendering.hpp"


extern PerfStats stats;

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i&   image_res,
                                 const Eigen::Vector3i&   map_size,
                                 const Eigen::Vector3f&   map_dim,
                                 const Eigen::Vector3f&   t_MW,
                                 std::vector<int>&        pyramid,
                                 const se::Configuration& config,
                                 const std::string        voxel_impl_yaml_path)
  : DenseSLAMSystem(image_res, map_size, map_dim,
      se::math::to_transformation(t_MW), pyramid, config, voxel_impl_yaml_path) {}

DenseSLAMSystem::DenseSLAMSystem(const Eigen::Vector2i&   image_res,
                                 const Eigen::Vector3i&   map_size,
                                 const Eigen::Vector3f&   map_dim,
                                 const Eigen::Matrix4f&   T_MW,
                                 std::vector<int>&        pyramid,
                                 const se::Configuration& config,
                                 const std::string        voxel_impl_yaml_path)
  : image_res_(image_res),
    depth_image_(image_res_.x(), image_res_.y()),
    rgba_image_(image_res_.x(), image_res_.y()),
    map_dim_(map_dim),
    map_size_(map_size),
    config_(config),
    init_T_MC_(T_MW),
    T_MC_(init_T_MC_),
    previous_T_MC_(T_MC_),
    iterations_(pyramid),
    reduction_output_(8 * 32, 0.0f),
    tracking_result_(image_res_.prod(), TrackData()),
    raycast_T_MC_(T_MC_),
    surface_point_cloud_M_(image_res_.x(), image_res_.y(), Eigen::Vector3f::Zero()),
    surface_normals_M_(image_res_.x(), image_res_.y(), Eigen::Vector3f::Zero()),
    render_T_MC_(&T_MC_),
    T_MW_(T_MW)
  {

    bool has_yaml_voxel_impl_config = false;
    YAML::Node yaml_voxel_impl_config = YAML::Load("");

    if (voxel_impl_yaml_path != "") {
      if (YAML::LoadFile(voxel_impl_yaml_path)["voxel_impl"]) {
        yaml_voxel_impl_config = YAML::LoadFile(voxel_impl_yaml_path)["voxel_impl"];
        has_yaml_voxel_impl_config = true;
      }
    }

    const float voxel_dim = map_dim_.x() / map_size_.x();
    if (has_yaml_voxel_impl_config) {
      VoxelImpl::configure(yaml_voxel_impl_config, voxel_dim);
    } else {
      VoxelImpl::configure(voxel_dim);
    }

    // Initialize the Gaussian for the bilateral filter
    constexpr int gaussian_size = gaussian_radius * 2 + 1;
    gaussian_.reserve(gaussian_size);
    for (int i = 0; i < gaussian_size; i++) {
      const int x = i - 2;
      gaussian_[i] = expf(-(x * x) / (2 * delta * delta));
    }

    // Initialize the scaled images
    for (unsigned int i = 0; i < iterations_.size(); ++i) {
      const int downsample = 1 << i;
      const Eigen::Vector2i res = image_res_ / downsample;
      scaled_depth_image_.emplace_back(res.x(), res.y(), 0.0f);
      input_point_cloud_C_.emplace_back(res.x(), res.y(), Eigen::Vector3f::Zero());
      input_normals_C_.emplace_back(res.x(), res.y(), Eigen::Vector3f::Zero());
    }

    // Initialize the map
    map_ = std::make_shared<se::Octree<VoxelImpl::VoxelType> >();
    map_->init(map_size_.x(), map_dim_.x());
}



bool DenseSLAMSystem::preprocessDepth(const float*           input_depth_image_data,
                                      const Eigen::Vector2i& input_depth_image_res,
                                      const bool             filter_depth){
  TICKD("preprocessDepth")
  downsampleDepthKernel(input_depth_image_data, input_depth_image_res, depth_image_);

  if (filter_depth) {
    bilateralFilterKernel(scaled_depth_image_[0], depth_image_, gaussian_,
        e_delta, gaussian_radius);
  } else {
    std::memcpy(scaled_depth_image_[0].data(), depth_image_.data(),
        sizeof(float) * image_res_.x() * image_res_.y());
  }
  TOCK("preprocessDepth")
  return true;
}



bool DenseSLAMSystem::preprocessColor(const uint32_t*        input_RGBA_image_data,
                                      const Eigen::Vector2i& input_RGBA_image_res) {

  TICKD("preprocessColor")
  downsampleImageKernel(input_RGBA_image_data, input_RGBA_image_res, rgba_image_);
  TOCK("preprocessColor")
  return true;
}



bool DenseSLAMSystem::track(const SensorImpl& sensor,
                            const float       icp_threshold) {

  TICK("TRACKING")
  // half sample the input depth maps into the pyramid levels
  for (unsigned int i = 1; i < iterations_.size(); ++i) {
    halfSampleRobustImageKernel(scaled_depth_image_[i], scaled_depth_image_[i - 1], e_delta * 3, 1);
  }

  // prepare the 3D information from the input depth maps
  for (unsigned int i = 0; i < iterations_.size(); ++i) {
    const float scaling_factor = 1.f / (1 << i);
    const SensorImpl scaled_sensor(sensor, scaling_factor);
    depthToPointCloudKernel(input_point_cloud_C_[i], scaled_depth_image_[i], scaled_sensor);
    if(sensor.left_hand_frame) {
      pointCloudToNormalKernel<true>(input_normals_C_[i], input_point_cloud_C_[i]);
    }
    else {
      pointCloudToNormalKernel<false>(input_normals_C_[i], input_point_cloud_C_[i]);
    }
  }

  previous_T_MC_ = T_MC_;

  for (int level = iterations_.size() - 1; level >= 0; --level) {
    Eigen::Vector2i reduction_output_res(
        image_res_.x() / (int) pow(2, level),
        image_res_.y() / (int) pow(2, level));
    for (int i = 0; i < iterations_[level]; ++i) {

      trackKernel(tracking_result_.data(), input_point_cloud_C_[level], input_normals_C_[level],
          surface_point_cloud_M_, surface_normals_M_, T_MC_, raycast_T_MC_, sensor, dist_threshold, normal_threshold);

      reduceKernel(reduction_output_.data(), reduction_output_res, tracking_result_.data(), image_res_);

      if (updatePoseKernel(T_MC_, reduction_output_.data(), icp_threshold))
        break;

    }
  }
  TOCK("TRACKING")
  return checkPoseKernel(T_MC_, previous_T_MC_, reduction_output_.data(),
      image_res_, track_threshold);
}



bool DenseSLAMSystem::integrate(const SensorImpl&  sensor,
                                const unsigned     frame) {

  TICK("INTEGRATION")
  const int num_blocks_per_pixel = map_->size()
    / ((VoxelBlockType::size_li));
  const size_t num_blocks_total = num_blocks_per_pixel
    * image_res_.x() * image_res_.y();
  allocation_list_.reserve(num_blocks_total);

  const Eigen::Matrix4f T_CM = se::math::to_inverse_transformation(T_MC_); // TODO:
  const size_t num_voxel = VoxelImpl::buildAllocationList(
      *map_,
      depth_image_,
      T_MC_,
      sensor,
      allocation_list_.data(),
      allocation_list_.capacity());

  if (num_voxel > 0) {
    TICKD("allocate")
    map_->allocate(allocation_list_.data(), num_voxel);
    TOCK("allocate")
  }

  VoxelImpl::integrate(
      *map_,
      depth_image_,
      T_CM,
      sensor,
      frame);
  TOCK("INTEGRATION")
  return true;
}



bool DenseSLAMSystem::raycast(const SensorImpl& sensor) {

  TICK("RAYCASTING")
  raycast_T_MC_ = T_MC_;
  raycastKernel<VoxelImpl>(*map_, surface_point_cloud_M_, surface_normals_M_,
      raycast_T_MC_, sensor);
  TOCK("RAYCASTING")
  return true;
}



void DenseSLAMSystem::renderVolume(uint32_t*              volume_RGBA_image_data,
                                   const Eigen::Vector2i& volume_RGBA_image_res,
                                   const SensorImpl&      sensor) {

  se::Image<Eigen::Vector3f> render_surface_point_cloud_M (image_res_.x(), image_res_.y());
  se::Image<Eigen::Vector3f> render_surface_normals_M (image_res_.x(), image_res_.y());
  if (render_T_MC_->isApprox(raycast_T_MC_)) {
    // Copy the raycast from the camera viewpoint. Can't safely use memcpy with
    // Eigen objects it seems.
    for (size_t i = 0; i < surface_point_cloud_M_.size(); ++i) {
      render_surface_point_cloud_M[i] = surface_point_cloud_M_[i];
      render_surface_normals_M[i] = surface_normals_M_[i];
    }
  } else {
    TICK("RAYCASTING")
    // Raycast the map from the render viewpoint.
    raycastKernel<VoxelImpl>(*map_, render_surface_point_cloud_M,
        render_surface_normals_M, *render_T_MC_, sensor);
    TOCK("RAYCASTING")
  }

  TICKD("renderVolume")
  renderVolumeKernel<VoxelImpl>(volume_RGBA_image_data, volume_RGBA_image_res,
      se::math::to_translation(*render_T_MC_), ambient,
      render_surface_point_cloud_M, render_surface_normals_M);
  TOCK("renderVolume")
}

void DenseSLAMSystem::renderTrack(uint32_t*              tracking_RGBA_image_data,
                                  const Eigen::Vector2i& tracking_RGBA_image_res) {

  TICKD("renderTrack")
  renderTrackKernel(tracking_RGBA_image_data, tracking_result_.data(), tracking_RGBA_image_res);
  TOCK("renderTrack")
}



void DenseSLAMSystem::renderDepth(uint32_t*              depth_RGBA_image_data,
                                  const Eigen::Vector2i& depth_RGBA_image_res,
                                  const SensorImpl&      sensor) {

  TICKD("renderDepth")
  renderDepthKernel(depth_RGBA_image_data, depth_image_.data(), depth_RGBA_image_res,
      sensor.near_plane, sensor.far_plane);
  TOCK("renderDepth")
}



void DenseSLAMSystem::renderRGBA(uint32_t*              output_RGBA_image_data,
                                 const Eigen::Vector2i& output_RGBA_image_res) {

  TICKD("renderRGBA")
  renderRGBAKernel(output_RGBA_image_data, output_RGBA_image_res, rgba_image_);
  TOCK("renderRGBA")
}



void DenseSLAMSystem::dumpMesh(const std::string filename, const bool print_path) {

  TICK("dumpMesh")
  if (print_path) {
    std::cout << "Saving triangle mesh to file :" << filename  << std::endl;
  }
  std::vector<se::Triangle> mesh;
  VoxelImpl::dumpMesh(*map_, mesh);
  if (str_utils::ends_with(filename, ".ply")) {
    save_mesh_ply(mesh, filename.c_str(), se::math::to_inverse_transformation(this->T_MW_));
  } else {
    save_mesh_vtk(mesh, filename.c_str(), se::math::to_inverse_transformation(this->T_MW_));
  }
  TOCK("dumpMesh")
}



void DenseSLAMSystem::saveStructure(const std::string base_filename) {

  TICK("saveStructure")
  std::stringstream f_s;
  f_s << base_filename << ".ply";
  se::save_octree_structure_ply(*map_, f_s.str().c_str());

  Eigen::Vector3i slice_coord = (map_->size() / map_->dim() * t_MC()).cast<int>();

  int scale = 0;
  // Save x plane
  std::stringstream fv_x;
  fv_x << base_filename << "_value_x.vtk";
  save_3d_value_slice_vtk(*map_, fv_x.str().c_str(),
                          Eigen::Vector3i(slice_coord.x(),     0,            0           ),
                          Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save y plane
  std::stringstream fv_y;
  fv_y << base_filename << "_value_y.vtk";
  save_3d_value_slice_vtk(*map_, fv_y.str().c_str(),
                          Eigen::Vector3i(0,            slice_coord.y(),     0           ),
                          Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save z plane
  std::stringstream fv_z;
  fv_z << base_filename << "_value_z.vtk";
  save_3d_value_slice_vtk(*map_, fv_z.str().c_str(),
                          Eigen::Vector3i(0,            0,            slice_coord.z()    ),
                          Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                          VoxelImpl::VoxelType::selectNodeValue, VoxelImpl::VoxelType::selectVoxelValue,
                          scale);

  // Save x plane
  std::stringstream fs_x;
  fs_x << base_filename << "_scale_x.vtk";
  save_3d_scale_slice_vtk(*map_, fs_x.str().c_str(),
                          Eigen::Vector3i(slice_coord.x(),     0,            0),
                          Eigen::Vector3i(slice_coord.x() + 1, map_->size(), map_->size()),
                          scale);

  // Save y plane
  std::stringstream fs_y;
  fs_y << base_filename << "_scale_y.vtk";
  save_3d_scale_slice_vtk(*map_, fs_y.str().c_str(),
                          Eigen::Vector3i(0, slice_coord.y(), 0),
                          Eigen::Vector3i(map_->size(), slice_coord.y() + 1, map_->size()),
                          scale);

  // Save z plane
  std::stringstream fs_z;
  fs_z << base_filename << "_scale_z.vtk";
  save_3d_scale_slice_vtk(*map_, fs_z.str().c_str(),
                          Eigen::Vector3i(0, 0, slice_coord.z()),
                          Eigen::Vector3i(map_->size(), map_->size(), slice_coord.z() + 1),
                          scale);
  TOCK("saveStructure")
}



void DenseSLAMSystem::structureStats(size_t& num_nodes,
                                     size_t& num_blocks,
                                     std::vector<size_t>& num_blocks_per_scale) {
  TICK("structureStats")
  num_nodes            = map_->pool().nodeBufferSize();
  num_blocks           = map_->pool().blockBufferSize();
  num_blocks_per_scale = map_->pool().blockBufferSizeDetailed();
  TOCK("structureStats")
}
