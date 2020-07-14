// SPDX-FileCopyrightText: 2020 Nils Funk, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause


#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <Eigen/Dense>
#include "se/octree.hpp"
#include "se/node.hpp"

/**
 * \brief Finds the first valid point along a ray starting from (ray_origin_M + t * ray_dir_M). Returns false if no
 * valid point can be found before the maximum travelled distance is reached.
 *
 * \param[in]     map                Reference to the octree
 * \param[in]     select_node_value  lambda function selecting the node value to be interpolated
 * \param[in]     select_voxel_value lambda function selecting the voxel value to be interpolated
 * \param[in]     ray_origin_M       Origin of the ray in map frame [m]
 * \param[in]     ray_dir_M          Direction of the ray in map frame [m]
 * \param[in]     step_size          Size of a step per iteration [m]
 * \param[in]     t_max              Maximum travelled distance along the ray [m]
 * \param[in,out] t                  Travelled distance along the ray [m]
 * \param[out]    f                  Interpolated value at ray_origin_M + t * ray_dir_M
 * \param[out]    p                  First valid point along the ray starting from
 *                                   the input t (ray_origin_M + t * ray_dir_M)
 * \return        is_valid           True if valid point could be found before reaching t_max. False otherwise.
 */
template <typename FieldType, template <typename FieldT> class OctreeT,
          typename NodeValueSelector,
          typename VoxelValueSelector>
bool find_valid_point(const OctreeT<FieldType>& map,
                      NodeValueSelector         select_node_value,
                      VoxelValueSelector        select_voxel_value,
                      const Eigen::Vector3f&    ray_origin_M,
                      const Eigen::Vector3f&    ray_dir_M,
                      const float               step_size,
                      const float               t_max,
                      float&                    t,
                      float&                    value,
                      Eigen::Vector3f&          point_M) {
  bool is_valid = false;
  Eigen::Vector3f ray_pos_M = ray_origin_M + t * ray_dir_M;
  value = map.interpAtPoint(ray_pos_M, select_node_value, select_voxel_value, 0, is_valid).first;
  while (!is_valid) {
    t += step_size;
    if (t > t_max) {
      return false;
    }
    ray_pos_M = ray_origin_M + t * ray_dir_M;
    value = map.interpAtPoint(ray_pos_M, select_node_value, select_voxel_value, 0, is_valid).first;
  }
  point_M = ray_pos_M;
  return true;
}

#endif //__COMMON_HPP
