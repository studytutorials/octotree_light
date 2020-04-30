// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2018 Nils Funk, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __OCTOMAP_IO_IMPL_HPP
#define __OCTOMAP_IO_IMPL_HPP

#if defined(SE_OCTOMAP) && SE_OCTOMAP

#include <cmath>
#include <fstream>
#include <iostream>

#include <Eigen/Dense>
#include <octomap/octomap.h>



template<typename VoxelT, typename FunctionT>
octomap::OcTree* se::to_octomap(const se::Octree<VoxelT>& octree,
                                const FunctionT           set_node_value) {

  // Test if the octree is initialized.
  if (octree.size() == 0) {
    return nullptr;
  }

  // supereight map parameters.
  const float voxel_dim = octree.dim() / octree.size();
  // The sample point of supereight voxels relative to the voxel
  // corner closest to the origin. Typically the sample point is
  // the voxel centre.
  const Eigen::Vector3f se_sample_offset = se::Octree<VoxelT>::sample_offset_frac_;
  // The sample point of octomap voxels relative to the voxel
  // corner closest to the origin. The sample point is the voxel
  // centre.
  const Eigen::Vector3f om_sample_offset = Eigen::Vector3f::Constant(0.5f);
  // Add this to supereight voxel coordinates to transform them to
  // OctoMap voxel coordinates.
  const Eigen::Vector3f se_to_om_offset = - se_sample_offset + om_sample_offset;

  // Initialize the octomap.
  octomap::OcTree* octomap = new octomap::OcTree(voxel_dim);

  se::Node<VoxelT>* node_stack[se::Octree<VoxelT>::max_voxel_depth * 8 + 1];
  size_t stack_idx = 0;

  se::Node<VoxelT>* node = octree.root();

  if (node != nullptr) {
    se::Node<VoxelT>* current = node;
    node_stack[stack_idx++] = current;

    while (stack_idx != 0) {
      node = current;

      if (node->isBlock()) {
        const se::VoxelBlock<VoxelT>* block
            = static_cast<se::VoxelBlock<VoxelT>*>(node);
        const Eigen::Vector3i block_coord = block->coordinates();
        const int block_size = static_cast<int>(se::VoxelBlock<VoxelT>::size);
        const int x_last = block_coord.x() + block_size;
        const int y_last = block_coord.y() + block_size;
        const int z_last = block_coord.z() + block_size;
        for (int z = block_coord.z(); z < z_last; ++z) {
          for (int y = block_coord.y(); y < y_last; ++y) {
            for (int x = block_coord.x(); x < x_last; ++x) {
              const Eigen::Vector3f voxel_coord_eigen
                  = voxel_dim * (Eigen::Vector3f(x, y, z) + se_to_om_offset);
              const octomap::point3d voxel_coord (
                  voxel_coord_eigen.x(),
                  voxel_coord_eigen.y(),
                  voxel_coord_eigen.z());
              const typename VoxelT::VoxelData voxel_data
                  = block->data(Eigen::Vector3i(x, y, z));
              set_node_value(*octomap, voxel_coord, voxel_data);
            }
          }
        }
      }

      if (node->children_mask_ == 0) {
        current = node_stack[--stack_idx];
        continue;
      }

      for (int child_idx = 0; child_idx < 8; ++child_idx) {
        se::Node<VoxelT>* child = node->child(child_idx);
        if (child != nullptr) {
          node_stack[stack_idx++] = child;
        }

      }
      current = node_stack[--stack_idx];
    }
  }

  // Make the octree consistent at all levels.
  octomap->updateInnerOccupancy();
  // Combine children with the same values.
  octomap->prune();
  // Return the pointer.
  return octomap;
}



template<typename VoxelT>
octomap::OcTree* se::to_octomap(const se::Octree<VoxelT>& octree) {

  // Create a lambda function to set the state of a single voxel.
  const auto set_node_value = [](octomap::OcTree&                  octomap,
                                 const octomap::point3d&           voxel_coord,
                                 const typename VoxelT::VoxelData& voxel_data) {
    // Just store the log-odds occupancy probability. Do not store log-odds of
    // 0 because OctoMap interprets it as occupied, whereas in supereight it
    // means unknown.
    if (voxel_data.x != 0.f) {
      octomap.setNodeValue(voxel_coord, voxel_data.x, true);
    }
  };

  // Convert to OctoMap and return a pointer.
  return se::to_octomap(octree, set_node_value);
}



template<typename VoxelT>
octomap::OcTree* se::to_binary_octomap(const se::Octree<VoxelT>& octree) {

  // Create a lambda function to set the state of a single voxel.
  const auto set_node_value = [](octomap::OcTree&                  octomap,
                                 const octomap::point3d&           voxel_coord,
                                 const typename VoxelT::VoxelData& voxel_data) {
    if (voxel_data.x > 0) {
      // Occupied
      octomap.updateNode(voxel_coord, true, true);
    } else if (voxel_data.x < 0) {
      // Free
      octomap.updateNode(voxel_coord, false, true);
    }
    // Do not update unknown voxels.
  };

  // Convert to OctoMap and return a pointer.
  return se::to_octomap(octree, set_node_value);
}

#endif

#endif

