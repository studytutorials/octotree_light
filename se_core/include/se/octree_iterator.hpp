// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __OCTREE_ITERATOR_HPP
#define __OCTREE_ITERATOR_HPP

#include <iterator>
#include <stack>

#include "node.hpp"
#include "volume.hpp"

namespace se {

  // Forward declare class
  template <typename T> class Octree;



  /** \brief Iterates over all valid data in the octree at the last scale it
   * was updated at.
   * The iterator performs a depth-first traversal of the octree. To use it
   * just use the se::Octree::begin() and se::Octree::end() functions or a
   * range-based for loop:
   *
   * ``` cpp
   * for (auto& volume : octree) {
   *     // Do stuff with volume
   * }
   * ```
   *
   * \note Changes to the se::Octree while iterating will result in strange
   * behavior.
   */
  template <typename T>
  class OctreeIterator {
    public:
      OctreeIterator(Octree<T>* octree);

      OctreeIterator(const OctreeIterator& other);

      OctreeIterator& operator++();

      OctreeIterator operator++(int);

      bool operator==(const OctreeIterator& other) const;

      bool operator!=(const OctreeIterator& other) const;

      Volume<T> operator*() const;

      // Iterator traits
      using difference_type = long;
      using value_type = Volume<T>;
      using pointer = Volume<T>*;
      using reference = Volume<T>&;
      using iterator_category = std::forward_iterator_tag;

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    private:
      typedef typename T::VoxelData VoxelData;

      // The 3 stacks should always be kept in sync.
      // Pointers to the Nodes that haven't been checked yet.
      std::stack<Node<T>*> node_stack_;
      // Pointers to the Nodes' parents.
      std::stack<Node<T>*> parent_stack_;
      // Child indexes of the Nodes with respect to their parents.
      std::stack<int> child_idx_stack_;
      // Only used for VoxelBlocks. The linear index of the next voxel to be
      // checked.
      int voxel_idx_;
      // Only used for VoxelBlocks. The scale of the next voxel to be checked.
      int voxel_scale_;
      // The edge length of a single voxel in metres.
      float voxel_dim_;
      // The current se::Volume. This is what is returned from
      // se::OctreeIterator::operator*.
      Volume<T> volume_;

      // Find the next Volume with valid data.
      void nextData();

      // Reset the iterator state to invalid. Used when finished iterating.
      void clear();
  };

} // namespace se

#include "octree_iterator_impl.hpp"

#endif // __OCTREE_ITERATOR_HPP

