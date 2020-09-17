// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __OCTREE_ITERATOR_IMPL_HPP
#define __OCTREE_ITERATOR_IMPL_HPP

#include <array>

template <typename T>
se::Volume<T>::Volume()
    : centre_M(Eigen::Vector3f::Constant(-1.0f)),
      dim(0.0f),
      size(0),
      data(T::initData()) {}



template <typename T>
se::Volume<T>::Volume(const Eigen::Vector3f& _centre_M,
                      float                  _dim,
                      int                    _size,
                      const VoxelData&       _data)
    : centre_M(_centre_M),
      dim(_dim),
      size(_size),
      data(_data) {}



template <typename T>
se::Volume<T>::Volume(const se::Volume<T>& other)
    : centre_M(other.centre_M),
      dim(other.dim),
      size(other.size),
      data(other.data) {}



template <typename T>
se::Volume<T>& se::Volume<T>::operator=(const se::Volume<T>& other) {
  centre_M = other.centre_M;
  dim = other.dim;
  size = other.size;
  data = other.data;
  return *this;
}



template <typename T>
bool se::Volume<T>::operator==(const se::Volume<T>& other) const {
  return (centre_M == other.centre_M)
      && (dim == other.dim)
      && (size == other.size)
      && (data == other.data);
}



template <typename T>
bool se::Volume<T>::operator!=(const se::Volume<T>& other) const {
  return !(*this == other);
}





template <typename T>
se::OctreeIterator<T>::OctreeIterator(se::Octree<T>* octree) {
  // Reset to an invalid (end) iterator
  clear();
  if (octree != nullptr) {
    Node<T>* root = octree->root();
    if (root != nullptr) {
      voxel_dim_ = octree->voxelDim();
      // Push the root's children on the stack
      for (int child_idx = 0; child_idx < 8; ++child_idx) {
        node_stack_.push(root->child(child_idx));
        parent_stack_.push(root);
        child_idx_stack_.push(child_idx);
      }
      // Find the next Volume
      nextData();
    }
  }
}



template <typename T>
se::OctreeIterator<T>::OctreeIterator(const se::OctreeIterator<T>& other)
  : node_stack_(other.node_stack_),
    parent_stack_(other.parent_stack_),
    child_idx_stack_(other.child_idx_stack_),
    voxel_idx_(other.voxel_idx_),
    voxel_scale_(other.voxel_scale_),
    voxel_dim_(other.voxel_dim_),
    volume_(other.volume_) {
}



template <typename T>
se::OctreeIterator<T>& se::OctreeIterator<T>::operator++() {
  if (node_stack_.empty()) {
    clear();
    return *this;
  }
  nextData();
  return *this;
}



template <typename T>
se::OctreeIterator<T> se::OctreeIterator<T>::operator++(int) {
  if (node_stack_.empty()) {
    clear();
    return *this;
  }
  se::OctreeIterator<T> previous_state (*this);
  nextData();
  return previous_state;
}



template <typename T>
bool se::OctreeIterator<T>::operator==(const se::OctreeIterator<T>& other) const {
  // The volume_ is obtained from the current place in the octree so checking
  // it would be redundant.
  return (node_stack_ == other.node_stack_)
      && (parent_stack_ == other.parent_stack_)
      && (child_idx_stack_ == other.child_idx_stack_)
      && (voxel_idx_ == other.voxel_idx_)
      && (voxel_scale_ == other.voxel_scale_)
      && (voxel_dim_ == other.voxel_dim_);
}



template <typename T>
bool se::OctreeIterator<T>::operator!=(const se::OctreeIterator<T>& other) const {
  return !(*this == other);
}



template <typename T>
se::Volume<T> se::OctreeIterator<T>::operator*() const {
  return volume_;
}



template <typename T>
void se::OctreeIterator<T>::nextData() {
  while (true) {
    if (node_stack_.empty()) {
      clear();
      return;
    }
    // Get the data from the top of the stacks
    Node<T>* node = node_stack_.top();
    Node<T>* parent = parent_stack_.top();
    const int child_idx = child_idx_stack_.top();
    if ((node != nullptr) && node->isBlock()) {
      // Allocated VoxelBlock
      VoxelBlock<T>* block = reinterpret_cast<VoxelBlock<T>*>(node);
      if (voxel_scale_ == -1) {
        // VoxelBlock encountered for the first time, initialize the scale and
        // voxel index
        voxel_idx_ = 0;
        voxel_scale_ = block->current_scale();
      }
      // Set the Volume data if valid
      const VoxelData& data = block->data(voxel_idx_, voxel_scale_);
      if (valid(data)) {
        const int size = block->scaleVoxelSize(voxel_scale_);
        const float dim = voxel_dim_ * size;
        const Eigen::Vector3i volume_coord = block->voxelCoordinates(voxel_idx_, voxel_scale_);
        const Eigen::Vector3f centre_M = voxel_dim_ * (volume_coord.cast<float>() + Eigen::Vector3f::Constant(size / 2.0f));
        volume_ = se::Volume<T>(centre_M, dim, size, data);
      }
      // Increment the voxel index to prepare for the next iteration
      voxel_idx_++;
      // Pop the node if all of its data has been read
      if (voxel_idx_ >= se::VoxelBlock<T>::scaleNumVoxels(voxel_scale_)) {
        node_stack_.pop();
        parent_stack_.pop();
        child_idx_stack_.pop();
        voxel_idx_ = -1;
        voxel_scale_ = -1;
      }
      // Return if a valid Volume was found, otherwise continue searching
      if (valid(data)) {
        return;
      }
    } else {
      // Pop the node since we'll be done with it after this call
      node_stack_.pop();
      parent_stack_.pop();
      child_idx_stack_.pop();
      if (node != nullptr) {
        // Non-leaf Node, push all children to the stack
        for (int child_idx = 0; child_idx < 8; child_idx++) {
          node_stack_.push(node->child(child_idx));
          parent_stack_.push(node);
          child_idx_stack_.push(child_idx);
        }
        // Then continue until a leaf is found
      } else {
        // Leaf Node
        const VoxelData& data = parent->childData(child_idx);
        if (valid(data)) {
          // Leaf Node with valid data
          const int size = parent->size() / 2;
          const float dim = voxel_dim_ * size;
          const Eigen::Vector3i node_centre_coord = parent->childCentreCoord(child_idx);
          const Eigen::Vector3f centre_M = voxel_dim_ * node_centre_coord.cast<float>();
          volume_ = se::Volume<T>(centre_M, dim, size, data);
          return;
        }
        // Ignore leaf Nodes without valid data
      }
    }
  }
}



template <typename T>
void se::OctreeIterator<T>::clear() {
  node_stack_ = std::stack<Node<T>*>();
  parent_stack_ = std::stack<Node<T>*>();
  child_idx_stack_ = std::stack<int>();
  voxel_idx_ = -1;
  voxel_scale_ = -1;
  voxel_dim_ = 0.0f;
  volume_ = se::Volume<T>();
}



template <typename T>
bool se::OctreeIterator<T>::valid(const VoxelData& data) {
  // TODO this should be just T::isValid(data) once that functionality is
  // available.
  return (data.y > 0);
}

#endif // __OCTREE_ITERATOR_IMPL_HPP

