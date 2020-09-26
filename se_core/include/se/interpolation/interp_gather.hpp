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

#ifndef INTERP_GATHER_H
#define INTERP_GATHER_H

#include "../octree_defines.h"
#include "../node.hpp"
#include "../octant_ops.hpp"

namespace se {
template<typename T>
using VoxelBlockType = typename T::VoxelBlockType;

namespace internal {
  constexpr int INVALID_SAMPLE = -2;

  /*
   * Interpolation's value gather offsets
   */
  static const Eigen::Vector3i interp_offsets[8] =
      {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
       {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};



  template <typename FieldType,
            typename NodeValueSelector,
            typename VoxelValueSelector,
            template<typename FieldT> class OctreeT,
            typename ValueT>
  inline void gather_local(const VoxelBlockType<FieldType>* block,
                           const OctreeT<FieldType>&        octree,
                           const Eigen::Vector3i&           base_coord,
                           const int                        scale,
                           const int                        stride,
                           NodeValueSelector                select_node_value,
                           VoxelValueSelector               select_voxel_value,
                           ValueT                           values[8]) {

    if (block) {
      values[0] = select_voxel_value(block->data(base_coord + stride * interp_offsets[0], scale));
      values[1] = select_voxel_value(block->data(base_coord + stride * interp_offsets[1], scale));
      values[2] = select_voxel_value(block->data(base_coord + stride * interp_offsets[2], scale));
      values[3] = select_voxel_value(block->data(base_coord + stride * interp_offsets[3], scale));
      values[4] = select_voxel_value(block->data(base_coord + stride * interp_offsets[4], scale));
      values[5] = select_voxel_value(block->data(base_coord + stride * interp_offsets[5], scale));
      values[6] = select_voxel_value(block->data(base_coord + stride * interp_offsets[6], scale));
      values[7] = select_voxel_value(block->data(base_coord + stride * interp_offsets[7], scale));
      return;
    }
    Eigen::Vector3i voxel_coord = base_coord + stride * interp_offsets[0];
    typename FieldType::VoxelData data;
    octree.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data);
    ValueT value = select_node_value(data);
    values[0] = value;
    values[1] = value;
    values[2] = value;
    values[3] = value;
    values[4] = value;
    values[5] = value;
    values[6] = value;
    values[7] = value;
    return;
  }



  template <typename FieldType, typename NodeValueSelector,
            typename VoxelValueSelector,
            template<typename FieldT> class OctreeT,
            typename ValueT>
  inline void gather_4(const VoxelBlockType<FieldType>* block,
                       const OctreeT<FieldType>&        octree,
                       const Eigen::Vector3i&           base_coord,
                       const int                        scale,
                       const int                        stride,
                       NodeValueSelector                select_node_value,
                       VoxelValueSelector               select_voxel_value,
                       const unsigned int               offsets[4],
                       ValueT                           values[8]) {

    if (block) {
      values[offsets[0]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[0]], scale));
      values[offsets[1]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[1]], scale));
      values[offsets[2]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[2]], scale));
      values[offsets[3]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[3]], scale));
      return;
    }
    Eigen::Vector3i voxel_coord = base_coord + stride * interp_offsets[offsets[0]];
    typename FieldType::VoxelData data;
    octree.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data);
    ValueT value = select_node_value(data);
    values[offsets[0]] = value;
    values[offsets[1]] = value;
    values[offsets[2]] = value;
    values[offsets[3]] = value;
    return;
  }



  template <typename FieldType,
            typename NodeValueSelector,
            typename VoxelValueSelector,
            template<typename FieldT> class OctreeT,
            typename ValueT>
  inline void gather_2(const VoxelBlockType<FieldType>* block,
                       const OctreeT<FieldType>&        octree,
                       const Eigen::Vector3i&           base_coord,
                       const int                        scale,
                       const int                        stride,
                       NodeValueSelector                select_node_value,
                       VoxelValueSelector               select_voxel_value,
                       const unsigned int               offsets[2],
                       ValueT                           values[8]) {

    if (block) {
      values[offsets[0]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[0]], scale));
      values[offsets[1]] = select_voxel_value(block->data(base_coord + stride * interp_offsets[offsets[1]], scale));
      return;
    }
    Eigen::Vector3i voxel_coord = base_coord + stride * interp_offsets[offsets[0]];
    typename FieldType::VoxelData data;
    octree.get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data);
    ValueT value = select_node_value(data);
    values[offsets[0]] = value;
    values[offsets[1]] = value;
    return;
  }

  template <typename FieldType, typename NodeValueSelector,
            typename VoxelValueSelector,
            template<typename FieldT> class OctreeT,
            typename ValueT>
  inline int gather_values(const OctreeT<FieldType>&  octree,
                           const Eigen::Vector3i&     base_coord,
                           const int                  scale,
                           NodeValueSelector          select_node_value,
                           VoxelValueSelector         select_voxel_value,
                           ValueT                     values[8]) {

    const int stride = 1 << scale;
    unsigned int block_size = VoxelBlockType<FieldType>::size_li;
    unsigned int crossmask
        = (((base_coord.x() & (block_size - 1)) == block_size - stride) << 2)
        | (((base_coord.y() & (block_size - 1)) == block_size - stride) << 1)
        |  ((base_coord.z() & (block_size - 1)) == block_size - stride);

    switch(crossmask) {
      case 0: /* all local */
        {
          VoxelBlockType<FieldType> * block = octree.fetch(base_coord.x(), base_coord.y(), base_coord(2));
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_local(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, values);
        }
        break;
      case 1: /* z crosses */
        {
          const unsigned int offs1[4] = {0, 1, 2, 3};
          const unsigned int offs2[4] = {4, 5, 6, 7};
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          const Eigen::Vector3i base_1_coord = base_coord + stride * interp_offsets[offs2[0]];
          block = octree.fetch(base_1_coord.x(), base_1_coord.y(), base_1_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
        }
        break;
      case 2: /* y crosses */
        {
          const unsigned int offs1[4] = {0, 1, 4, 5};
          const unsigned int offs2[4] = {2, 3, 6, 7};
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          if (block && block->current_scale() > scale)
            return block->current_scale();
          const Eigen::Vector3i base_1_coord = base_coord + stride * interp_offsets[offs2[0]];
          block = octree.fetch(base_1_coord.x(), base_1_coord.y(), base_1_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
        }
        break;
      case 3: /* y, z cross */
        {
          const unsigned int offs1[2] = {0, 1};
          const unsigned int offs2[2] = {2, 3};
          const unsigned int offs3[2] = {4, 5};
          const unsigned int offs4[2] = {6, 7};
          const Eigen::Vector3i base_2_coord = base_coord + stride * interp_offsets[offs2[0]];
          const Eigen::Vector3i base_3_coord = base_coord + stride * interp_offsets[offs3[0]];
          const Eigen::Vector3i base_4_coord = base_coord + stride * interp_offsets[offs4[0]];
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          block = octree.fetch(base_2_coord.x(), base_2_coord.y(), base_2_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
          block = octree.fetch(base_3_coord.x(), base_3_coord.y(), base_3_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs3, values);
          block = octree.fetch(base_4_coord.x(), base_4_coord.y(), base_4_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs4, values);
        }
        break;
      case 4: /* x crosses */
        {
          const unsigned int offs1[4] = {0, 2, 4, 6};
          const unsigned int offs2[4] = {1, 3, 5, 7};
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          const Eigen::Vector3i base_1_coord = base_coord + stride * interp_offsets[offs2[0]];
          block = octree.fetch(base_1_coord.x(), base_1_coord.y(), base_1_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_4(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
        }
        break;
      case 5: /* x,z cross */
        {
          const unsigned int offs1[2] = {0, 2};
          const unsigned int offs2[2] = {1, 3};
          const unsigned int offs3[2] = {4, 6};
          const unsigned int offs4[2] = {5, 7};
          const Eigen::Vector3i base_2_coord = base_coord + stride * interp_offsets[offs2[0]];
          const Eigen::Vector3i base_3_coord = base_coord + stride * interp_offsets[offs3[0]];
          const Eigen::Vector3i base_4_coord = base_coord + stride * interp_offsets[offs4[0]];
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          block = octree.fetch(base_2_coord.x(), base_2_coord.y(), base_2_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
          block = octree.fetch(base_3_coord.x(), base_3_coord.y(), base_3_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs3, values);
          block = octree.fetch(base_4_coord.x(), base_4_coord.y(), base_4_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs4, values);
        }
        break;
      case 6: /* x,y cross */
        {
          const unsigned int offs1[2] = {0, 4};
          const unsigned int offs2[2] = {1, 5};
          const unsigned int offs3[2] = {2, 6};
          const unsigned int offs4[2] = {3, 7};
          const Eigen::Vector3i base_2_coord = base_coord + stride * interp_offsets[offs2[0]];
          const Eigen::Vector3i base_3_coord = base_coord + stride * interp_offsets[offs3[0]];
          const Eigen::Vector3i base_4_coord = base_coord + stride * interp_offsets[offs4[0]];
          VoxelBlockType<FieldType>* block = octree.fetch(base_coord.x(), base_coord.y(), base_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs1, values);
          block = octree.fetch(base_2_coord.x(), base_2_coord.y(), base_2_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs2, values);
          block = octree.fetch(base_3_coord.x(), base_3_coord.y(), base_3_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs3, values);
          block = octree.fetch(base_4_coord.x(), base_4_coord.y(), base_4_coord.z());
          if (block && block->current_scale() > scale)
            return block->current_scale();
          gather_2(block, octree, base_coord, scale, stride, select_node_value, select_voxel_value, offs4, values);
        }
        break;

      case 7:
        {
          Eigen::Vector3i voxels_coord[8];
          voxels_coord[0] = base_coord + stride * interp_offsets[0];
          voxels_coord[1] = base_coord + stride * interp_offsets[1];
          voxels_coord[2] = base_coord + stride * interp_offsets[2];
          voxels_coord[3] = base_coord + stride * interp_offsets[3];
          voxels_coord[4] = base_coord + stride * interp_offsets[4];
          voxels_coord[5] = base_coord + stride * interp_offsets[5];
          voxels_coord[6] = base_coord + stride * interp_offsets[6];
          voxels_coord[7] = base_coord + stride * interp_offsets[7];

          for (int i = 0; i < 8; ++i) {
            auto block = octree.fetch(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z());

            if (block && block->current_scale() > scale)
              return block->current_scale();

            if (block) {
              values[i] = select_voxel_value(block->data(voxels_coord[i], scale));
            } else {
              typename FieldType::VoxelData data;
              octree.get(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z(), data);
              values[i] = select_node_value(data);
            }
          }
        }
        break;
    }
    return scale;
  }

  template <typename FieldType,
      typename ValueSelector,
      template<typename FieldT> class OctreeT,
      typename ValueT>
  inline int gather_values(const OctreeT<FieldType>&  octree,
                           const Eigen::Vector3i&     base_coord,
                           const int                  scale,
                           ValueSelector              select_value,
                           ValueT                     values[8]) {
    return gather_values(octree, base_coord, scale, select_value, select_value, values);
  }

  /**
   * \brief Fetch the node value corresponding to the octant neighbour along the
   * specified direction.
   *
   * \param[in] stack             The stack of ancestor nodes of octant
   * \param[in] octant            The base_coord octant.
   * \param[in] max_depth         The maximum depth of the tree.
   * \param[in] dir               The direction along which to fetch the neighbour.
   * \param[in] select_node_value Lambda function selecting the node's value.
   *
   * \note Only positive search directions are allowed along any axes.
   *
   * \return The siblings note value. If the search fails the second element of the returned
   *         is set to Eigen::Vector3i::Constant(INVALID_SAMPLE).
   */
  template <typename Precision, typename FieldType, typename NodeValueSelector>
  static inline std::pair<Precision, Eigen::Vector3i> fetch_neighbour_sample(
      Node<FieldType>*  stack[],
      Node<FieldType>*  octant,
      const int         max_depth,
      const int         dir,
      NodeValueSelector select_node_value) {

    int depth = se::keyops::depth(octant->code());
    while (depth > 0) {
      int child_idx = se::child_idx(stack[depth]->code(), max_depth);
      int sibling = child_idx ^ dir;
      if ((sibling & dir) == dir) { // if sibling still in octant's family
        const int child_size = 1 << (max_depth - depth);
        const Eigen::Vector3i coords = se::keyops::decode(stack[depth-1]->code())
            + child_size * Eigen::Vector3i((sibling & 1), (sibling & 2) >> 1, (sibling & 4) >> 2);
        return {select_node_value(stack[depth - 1]->childData(sibling)), coords};
      }
      depth--;
    }
    return {Precision(), Eigen::Vector3i::Constant(INVALID_SAMPLE)};
  }



  /**
   * \brief Fetch the neighbour of octant in the desired direction which is at
   * most refined as the starting octant.
   *
   * \param[in] stack     The stack of ancestor nodes of octant
   * \param[in] octant    The base_coord octant.
   * \param[in] max_depth The maximum depth of the tree.
   * \param[in] dir       The direction along which to fetch the neighbour.
   *
   * \note Only positive search directions are allowed along any axes.
   *
   * \result The siblings node pointer. If the search fails a nullptr is returned.
   */
  template <typename FieldType>
  static inline Node<FieldType>* fetch_neighbour(Node<FieldType>* stack[],
                                                 Node<FieldType>* octant,
                                                 const int        max_depth,
                                                 const int        dir) {

    int depth = se::keyops::depth(octant->code());
    while (depth > 0) {
      int child_idx = se::child_idx(stack[depth]->code(), max_depth);
      int sibling = child_idx ^ dir;
      if ((sibling & dir) == dir) { // if sibling still in octant's family
        return stack[depth - 1]->child(sibling);
      }
      depth--;
    }
    return nullptr;
  }



  /*! \brief Fetch the finest octant containing (x,y,z) starting from root node.
   * It is required that pos is contained withing the root node, i.e. pos is
   * within the interval [root.pos, root.pos + root.size].
   * \param stack stack of traversed nodes
   * \param root Root node from where the search starts.
   * \param pos integer position of searched octant
   */
  template <typename T>
  static inline Node<T>* fetch(Node<T>*               stack[],
                               Node<T>*               root,
                               const int              max_depth,
                               const Eigen::Vector3i& node_coord) {

    unsigned node_size = (1 << (max_depth - se::keyops::depth(root->code()))) / 2;
    constexpr unsigned int block_size = BLOCK_SIZE;
    Node<T>* node = root;
    int d = 0;
    for (; node_size >= block_size; ++d, node_size = node_size >> 1) {
      stack[d] = node;
      auto next = node->child(
          (node_coord.x() & node_size) > 0u,
          (node_coord.y() & node_size) > 0u,
          (node_coord.z() & node_size) > 0u);
      if (!next)
        break;
      node = next;
    }
    stack[d] = node;
    return node;
  }
} // end namespace internal
} // end namespace se

#endif

