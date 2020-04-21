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
#ifndef OCTREE_COLLISION_HPP
#define OCTREE_COLLISION_HPP
#include "../node.hpp"
#include "../octree.hpp"
#include "aabb_collision.hpp"

namespace se {
namespace geometry {
enum class collision_status {
  occupied,
  unseen,
  empty
};

/*! \brief Implements a simple state machine to update the collision status.
 * The importance order is given as follows in ascending order:
 * Empty, Unseen, Occupied.
 * \param previous_status
 * \param new_status
 */
inline collision_status update_status(const collision_status previous_status,
    const collision_status new_status) {
  switch(previous_status) {
    case collision_status::unseen:
      if(new_status != collision_status::occupied)
        return previous_status;
      else
        return new_status;
      break;
    case collision_status::occupied:
      return previous_status;
      break;
    default:
      return new_status;
      break;
  }
}

/*! \brief Perform a collision test for each voxel value in the input voxel
 * block. The test function test takes as input a voxel value and returns a
 * collision_status. This is used to distinguish between seen-empty voxels and
 * occupied voxels.
 * \param block voxel block of type FieldType
 * \param test function that takes a voxel and returns a collision_status value
 */
template <typename FieldType, typename TestVoxelF>
collision_status collides_with(const se::VoxelBlock<FieldType>* block,
    const Eigen::Vector3i bbox_coord, const Eigen::Vector3i size, TestVoxelF test) {
  collision_status status = collision_status::empty;
  const Eigen::Vector3i block_coord = block->coordinates();
  int x, y, z, block_size;
  block_size = (int) se::VoxelBlock<FieldType>::size;
  int x_last = block_coord.x() + block_size;
  int y_last = block_coord.y() + block_size;
  int z_last = block_coord.z() + block_size;
  for(z = block_coord.z(); z < z_last; ++z){
    for (y = block_coord.y(); y < y_last; ++y){
      for (x = block_coord.x(); x < x_last; ++x){

        typename se::VoxelBlock<FieldType>::VoxelData data;
        const Eigen::Vector3i voxel_coord{x, y, z};
        if(!geometry::aabb_aabb_collision(bbox_coord, size,
          voxel_coord, Eigen::Vector3i::Constant(1))) continue;
        data = block->data(Eigen::Vector3i(x, y, z));
        status = update_status(status, test(data));
      }
    }
  }
  return status;
}

/*! \brief Perform a collision test between the input octree map and the
 * input axis aligned bounding box bbox_coord of extension size. The test function
 * test takes as input a voxel data and returns a collision_status. This is
 * used to distinguish between seen-empty voxels and occupied voxels.
 * \param octree octree map
 * \param bbox_coord test bounding box lower bottom corner
 * \param size extension in number of voxels of the bounding box
 * \param test function that takes a voxel and returns a collision_status data
 */

template <typename FieldType, typename TestVoxelF>
collision_status collides_with(const Octree<FieldType>& octree,
    const Eigen::Vector3i bbox_coord, const Eigen::Vector3i bbox_size, TestVoxelF test) {

  typedef struct stack_entry {
    se::Node<FieldType>* node_ptr;
    Eigen::Vector3i coordinates;
    int size;
    typename se::Node<FieldType>::VoxelData parent_data;
  } stack_entry;

  stack_entry stack[Octree<FieldType>::max_voxel_depth * 8 + 1];
  size_t stack_idx = 0;

  se::Node<FieldType>* node = octree.root();
  if(!node) return collision_status::unseen;

  stack_entry current;
  current.node_ptr = node;
  current.size = octree.size();
  current.coordinates = {0, 0, 0};
  stack[stack_idx++] = current;
  collision_status status = collision_status::empty;

  while(stack_idx != 0){
    node = current.node_ptr;

    if(node->isBlock()){
      status = collides_with(static_cast<se::VoxelBlock<FieldType>*>(node),
          bbox_coord, bbox_size, test);
    }

    if(node->children_mask_ == 0) {
       current = stack[--stack_idx];
       continue;
    }

    for(int child_idx = 0; child_idx < 8; ++child_idx){
      se::Node<FieldType>* child = node->child(child_idx);
      stack_entry child_descr;
      child_descr.node_ptr = nullptr;
      child_descr.size = current.size / 2;
      child_descr.coordinates =
        Eigen::Vector3i(current.coordinates.x() + child_descr.size*((child_idx & 1) > 0),
            current.coordinates.y() + child_descr.size*((child_idx & 2) > 0),
            current.coordinates.z() + child_descr.size*((child_idx & 4) > 0));

      const bool overlaps = geometry::aabb_aabb_collision(bbox_coord, bbox_size,
          child_descr.coordinates, Eigen::Vector3i::Constant(child_descr.size));

      if(overlaps && child != nullptr) {
        child_descr.node_ptr = child;
        child_descr.parent_data = node->data_[0];
        stack[stack_idx++] = child_descr;
      } else if(overlaps && child == nullptr) {
        status = update_status(status, test(node->data_[0]));
      }
    }
    current = stack[--stack_idx];
  }
  return status;
}
}
}
#endif
