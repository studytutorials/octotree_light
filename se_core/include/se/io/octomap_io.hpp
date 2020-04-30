// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2018 Nils Funk, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

/*! \file
 * Functions to convert an se::Octree to an OctoMap octomap::OcTree.
 * For documentation on octomap::OcTree see
 * http://octomap.github.io/octomap/doc/classoctomap_1_1OcTree.html
 *
 * \warning These functions do not handle multiresolution octrees correctly.
 * There will be missing or out of date data when saving them. See the TODO
 * list for how to fix this.
 *
 * \todo
 * - Test the content of the individual voxels in the unit tests. Octomap does
 *   not seem to provide an easy way to access information at the voxel level.
 * - Handle leaf Nodes with data. Currently only VoxelBlocks are saved while
 *   MultiresOFusion can store information inside leaf Nodes. For each node
 *   without children, iterate over the coordinates of all its children and
 *   save the node data.
 * - Propagate information integrated at higher scales to the voxel scale. The
 *   information is not propagated to the finest scale before saving which will
 *   be an issue for both MultiresOFusion and MultiresTSDF. It should call the
 *   VoxelImpl method to make the map consistent at all levels.
 * - Test both of the above cases in the unit tests.
 */

#ifndef __OCTOMAP_IO_HPP
#define __OCTOMAP_IO_HPP

#if defined(SE_OCTOMAP) && SE_OCTOMAP

#include <cstring>

#include <octomap/octomap.h>

#include "se/octree.hpp"



namespace se {

  /*! Convert an se::Octree to an OctoMap.
   * A lambda function can be used to specify exactly how the se::Octree data
   * is converted to OctoMap data. The signature of the lambda function
   * should be
   * ```
   * [](octomap::OcTree& octomap, const octomap::point3d& voxel_coord, const VoxelT::VoxelData& voxel_data)
   * ```
   * where `octomap` is the OctoMap format octree being created, `voxel_coord`
   * the coordinates of the current voxel in meters and `voxel_data` the data
   * stored in the se::Octree for this voxel. The lambda function should call
   * ```
   * octomap.setNodeValue(voxel_coord, new_value);
   * ```
   * to set the OctoMap voxel occupancy value to `new_value`. Keep in mind that
   * OctoMap only stores the occupancy probability in log-odds in the voxel
   * whereas supereight stores more data.
   *
   * \note To save the OctoMap to a file, call the
   * `writeBinary(const std::string& filename)` method of the returned
   * OctoMap. Make sure the extension of the provided filename is `.bt`.
   *
   * \warning This function does not handle multiresolution octrees correctly.
   * There will be missing or out of date data when saving them.
   *
   * \param[in] octree         The octree to convert.
   * \param[in] set_node_value The lambda function used to convert the octree
   *                           data.
   * \return A pointer to an OctoMap, allocated by `new`. Wrap it in a
   *         smart pointer or manually deallocate the memory with
   *         `delete` after use. Returns nullptr if octree has not been
   *         properly initialized.
   */
  template<typename VoxelT, typename FunctionT>
  octomap::OcTree* to_octomap(const se::Octree<VoxelT>& octree,
                              const FunctionT           set_node_value);



  /*! Convert an se::Octree to an OctoMap.
   * The log-odds occupancy probability of the se::Octree is saved directly in
   * the OctoMap.
   *
   * This function is just a wrapper around
   * se::to_octomap(const se::Octree<VoxelT>&, const FunctionT).
   *
   * \note To save the OctoMap to a file, call the
   * `writeBinary(const std::string& filename)` method of the returned
   * OctoMap. Make sure the extension of the provided filename is `.bt`.
   *
   * \warning This function does not handle multiresolution octrees correctly.
   * There will be missing or out of date data when saving them.
   *
   * \param[in] octree The octree to convert.
   * \return A pointer to an OctoMap, allocated by `new`. Wrap it in a
   *         smart pointer or manually deallocate the memory with
   *         `delete` after use. Returns nullptr if octree has not been
   *         properly initialized.
   */
  template<typename VoxelT>
  octomap::OcTree* to_octomap(const se::Octree<VoxelT>& octree);



  /*! Convert an se::Octree to an OctoMap.
   * Only a binary state, occupied or free, is saved in the OctoMap.
   * Voxels are considered occupied if their occupancy probability is
   * greater than 0.5, free if it is smaller than 0.5 and unknown if it
   * is exactly 0.5. Unknown voxels are not saved at all in the OctoMap.
   *
   * This function is just a wrapper around
   * se::to_octomap(const se::Octree<VoxelT>&, const FunctionT).
   *
   * \note To save the OctoMap to a file, call the
   * `writeBinary(const std::string& filename)` method of the returned
   * OctoMap. Make sure the extension of the provided filename is `.bt`.
   *
   * \warning This function does not handle multiresolution octrees correctly.
   * There will be missing or out of date data when saving them.
   *
   * \param[in] octree The octree to convert.
   * \return A pointer to an OctoMap, allocated by `new`. Wrap it in a
   *         smart pointer or manually deallocate the memory with
   *         `delete` after use. Returns nullptr if octree has not been
   *         properly initialized.
   */
  template<typename VoxelT>
  octomap::OcTree* to_binary_octomap(const se::Octree<VoxelT>& octree);

} // namespace se

#include "octomap_io_impl.hpp"

#endif

#endif

