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

#ifndef NODE_IMPL_HPP
#define NODE_IMPL_HPP

#include <algorithm>
#include <cstdlib>

#include "se/octant_ops.hpp"

namespace se {
// Node implementation

template <typename T>
Node<T>::Node(const typename T::VoxelData init_data) :
    code_(0),
    size_(0),
    children_mask_(0),
    timestamp_(0) {
  for (unsigned int child_idx = 0; child_idx < 8; child_idx++) {
    children_data_[child_idx] = init_data;
    children_ptr_[child_idx]  = nullptr;
    parent_ptr_               = nullptr;
  }
}

template <typename T>
Node<T>::Node(const Node<T>& node) {
  initFromNode(node);
}

template <typename T>
void Node<T>::operator=(const Node<T>& node) {
  initFromNode(node);
}

template <typename T>
const typename T::VoxelData& Node<T>::data() const {
  // Used so we can return a reference to invalid data
  static const typename T::VoxelData invalid = T::invalid();
  // Return invalid data if this is the root node
  if (parent_ptr_ == nullptr) {
    return invalid;
  }
  // Find the child index of this node in its parent
  // TODO this is not the most efficient way but computing the child index directly requires the
  // octree dimensions
  int child_idx = 0;
  for (; child_idx < 8; child_idx++) {
    if (parent_ptr_->child(child_idx) == this) {
      break;
    }
  }
  if (child_idx < 8) {
    return parent_ptr_->childData(child_idx);
  } else {
    // The parent does not contain a pointer to this node, broken octree
    return invalid;
  }
}

template <typename T>
void Node<T>::initFromNode(const se::Node<T>& node) {
  code_           = node.code();
  size_           = node.size_;
  children_mask_  = node.children_mask();
  timestamp_      = node.timestamp();
  active_         = node.active();
  std::copy(node.childrenData(), node.childrenData() + 8, children_data_);
}

template <typename T>
Eigen::Vector3i Node<T>::coordinates() const {
  return se::keyops::decode(code_);
}

template <typename T>
Eigen::Vector3i Node<T>::centreCoordinates() const {
  return coordinates() + Eigen::Vector3i::Constant(size_ / 2);
}

template <typename T>
Eigen::Vector3i Node<T>::childCoord(const int child_idx) const {
  const std::div_t d1 = std::div(child_idx, 4);
  const std::div_t d2 = std::div(d1.rem, 2);
  const int rel_z = d1.quot;
  const int rel_y = d2.quot;
  const int rel_x = d2.rem;
  const int child_size = size_ / 2;
  return coordinates() + child_size * Eigen::Vector3i(rel_x, rel_y, rel_z);
}

template <typename T>
Eigen::Vector3i Node<T>::childCentreCoord(const int child_idx) const {
  const int child_size = size_ / 2;
  return childCoord(child_idx) + Eigen::Vector3i::Constant(child_size / 2);
}



// Voxel block base implementation

template <typename T>
VoxelBlock<T>::VoxelBlock(const int current_scale,
                          const int min_scale) :
    coordinates_(Eigen::Vector3i::Constant(0)),
    current_scale_(current_scale),
    min_scale_(min_scale) {}

template <typename T>
VoxelBlock<T>::VoxelBlock(const VoxelBlock<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlock<T>::operator=(const VoxelBlock<T>& block) {
  initFromBlock(block);
}

template <typename T>
Eigen::Vector3i VoxelBlock<T>::voxelCoordinates(const int voxel_idx) const {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = scaleNumVoxels(scale);
  }
  return voxelCoordinates(remaining_voxel_idx, scale);
}

template <typename T>
Eigen::Vector3i VoxelBlock<T>::voxelCoordinates(const int voxel_idx, const int scale) const {
  const std::div_t d1 = std::div(voxel_idx, se::math::sq(scaleSize(scale)));
  const std::div_t d2 = std::div(d1.rem, scaleSize(scale));
  const int z = d1.quot;
  const int y = d2.quot;
  const int x = d2.rem;
  return this->coordinates_ + scaleVoxelSize(scale) * Eigen::Vector3i(x, y, z);
}

template <typename T>
constexpr int VoxelBlock<T>::scaleSize(const int scale) {
  return size_li >> scale;
}

template <typename T>
constexpr int VoxelBlock<T>::scaleVoxelSize(const int scale) {
  return 1 << scale;
}

template <typename T>
constexpr int VoxelBlock<T>::scaleNumVoxels(const int scale) {
  return se::math::cu(scaleSize(scale));
}

template <typename T>
constexpr int VoxelBlock<T>::scaleOffset(const int scale) {
  int scale_offset = 0;
  for (int s = 0; s < scale; ++s) {
    scale_offset += scaleNumVoxels(s);
  }
  return scale_offset;
}

template <typename T>
void VoxelBlock<T>::initFromBlock(const VoxelBlock<T>& block) {
  this->code_          = block.code();
  this->size_          = block.size_;
  this->children_mask_ = block.children_mask();
  this->timestamp_     = block.timestamp();
  this->active_        = block.active();
  coordinates_   = block.coordinates();
  min_scale_     = block.min_scale();
  current_scale_ = block.current_scale();
  std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
}



// Voxel block finest scale allocation implementation

template <typename T>
VoxelBlockFinest<T>::VoxelBlockFinest(const typename T::VoxelData init_data) : VoxelBlock<T>(0, 0) {
  for (unsigned int voxel_idx = 0; voxel_idx < num_voxels_in_block; voxel_idx++) {
    block_data_[voxel_idx] = init_data;
  }
}

template <typename T>
VoxelBlockFinest<T>::VoxelBlockFinest(const VoxelBlockFinest<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlockFinest<T>::operator=(const VoxelBlockFinest<T>& block) {
  initFromBlock(block);
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const Eigen::Vector3i& voxel_coord) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  return block_data_[voxel_offset.x() +
                     voxel_offset.y() * this->size_li +
                     voxel_offset.z() * this->size_sq];
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const Eigen::Vector3i& voxel_coord,
                          const int              /* scale */) const {
  return data(voxel_coord);
}

template <typename T>
inline void VoxelBlockFinest<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const VoxelData&       voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[voxel_offset.x() +
              voxel_offset.y() * this->size_li +
              voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockFinest<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const int              /* scale */,
                                         const VoxelData&       voxel_data){
  setData(voxel_coord, voxel_data);
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const int voxel_idx) const {
  return block_data_[voxel_idx];
}

template <typename T>
inline void VoxelBlockFinest<T>::setData(const int voxel_idx, const VoxelData& voxel_data) {
  block_data_[voxel_idx] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFinest<T>::data(const int voxel_idx, const int /* scale */) const {
  return block_data_[voxel_idx];
}

template <typename T>
inline void VoxelBlockFinest<T>::setData(const int        voxel_idx,
                                         const int        /* scale */,
                                         const VoxelData& voxel_data) {
  block_data_[voxel_idx] = voxel_data;
}

template <typename T>
void VoxelBlockFinest<T>::initFromBlock(const VoxelBlockFinest<T>& block) {
  this->code_          = block.code();
  this->size_          = block.size_;
  this->children_mask_ = block.children_mask();
  this->timestamp_     = block.timestamp();
  this->active_        = block.active();
  this->coordinates_   = block.coordinates();
  this->min_scale_     = block.min_scale();
  this->current_scale_ = block.current_scale();
  std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
  std::copy(block.blockData(), block.blockData() + num_voxels_in_block, blockData());
}



// Voxel block full scale allocation implementation

template <typename T>
VoxelBlockFull<T>::VoxelBlockFull(const typename T::VoxelData init_data) : VoxelBlock<T>(0, -1) {
  for (unsigned int voxel_idx = 0; voxel_idx < num_voxels_in_block; voxel_idx++) {
    block_data_[voxel_idx] = init_data;
  }
}

template <typename T>
VoxelBlockFull<T>::VoxelBlockFull(const VoxelBlockFull<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlockFull<T>::operator=(const VoxelBlockFull<T>& block) {
  initFromBlock(block);
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  return block_data_[voxel_offset.x() +
                     voxel_offset.y() * this->size_li +
                     voxel_offset.z() * this->size_sq];
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const Eigen::Vector3i& voxel_coord, const int scale) const {
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = this->size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }
  const int local_size = this->size_li / (1 << scale);
  voxel_offset = voxel_offset / (1 << scale);
  return block_data_[scale_offset + voxel_offset.x() +
                     voxel_offset.y() * local_size +
                     voxel_offset.z() * se::math::sq(local_size)];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord,
                                       const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[voxel_offset.x() +
              voxel_offset.y() * this->size_li +
              voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const Eigen::Vector3i& voxel_coord, const int scale,
                                   const VoxelData& voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  int scale_offset = 0;
  int scale_tmp = 0;
  int num_voxels = this->size_cu;
  while(scale_tmp < scale) {
    scale_offset += num_voxels;
    num_voxels /= 8;
    ++scale_tmp;
  }

  const int size_at_scale = this->size_li / (1 << scale);
  voxel_offset = voxel_offset / (1 << scale);
  block_data_[scale_offset + voxel_offset.x() +
              voxel_offset.y() * size_at_scale +
              voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const int voxel_idx) const {
  return block_data_[voxel_idx];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const int voxel_idx, const VoxelData& voxel_data){
  block_data_[voxel_idx] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockFull<T>::data(const int voxel_idx, const int scale) const {
  return block_data_[this->scaleOffset(scale) + voxel_idx];
}

template <typename T>
inline void VoxelBlockFull<T>::setData(const int        voxel_idx,
                                       const int        scale,
                                       const VoxelData& voxel_data) {
  block_data_[this->scaleOffset(scale) + voxel_idx] = voxel_data;
}

template <typename T>
void VoxelBlockFull<T>::initFromBlock(const VoxelBlockFull<T>& block) {
  this->code_          = block.code();
  this->size_          = block.size_;
  this->children_mask_ = block.children_mask();
  this->timestamp_     = block.timestamp();
  this->active_        = block.active();
  this->coordinates_   = block.coordinates();
  this->min_scale_     = block.min_scale();
  this->current_scale_ = block.current_scale();
  std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
  std::copy(block.blockData(), block.blockData() + num_voxels_in_block, blockData());
}



// Voxel block single scale allocation implementation

template <typename T>
VoxelBlockSingle<T>::VoxelBlockSingle(const typename T::VoxelData init_data)
    : VoxelBlock<T>(0, -1) , init_data_(init_data) {}

template <typename T>
VoxelBlockSingle<T>::VoxelBlockSingle(const VoxelBlockSingle<T>& block) {
  initFromBlock(block);
}

template <typename T>
void VoxelBlockSingle<T>::operator=(const VoxelBlockSingle<T>& block) {
  initFromBlock(block);
}

template <typename T>
VoxelBlockSingle<T>::~VoxelBlockSingle() {
  for (auto data_at_scale : block_data_) {
    delete[] data_at_scale;
  }
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::initData() const { return init_data_; }

template <typename T>
inline void VoxelBlockSingle<T>::setInitData(const VoxelData& init_data) { init_data_ = init_data;}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord) const {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
    return init_data_;
  } else {
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    return block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                                 voxel_offset.y() * this->size_li +
                                                 voxel_offset.z() * this->size_sq];
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const VoxelData&       voxel_data){
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                        voxel_offset.y() * this->size_li +
                                        voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const VoxelData&       voxel_data){
  allocateDownTo(0);
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  block_data_[VoxelBlock<T>::max_scale][voxel_offset.x() +
                                        voxel_offset.y() * this->size_li +
                                        voxel_offset.z() * this->size_sq] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const Eigen::Vector3i& voxel_coord,
                          const int              scale) const {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
    return init_data_;
  } else {
    Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
    voxel_offset = voxel_offset / (1 << scale);
    const int size_at_scale = this->size_li >> scale;
    return block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                                         voxel_offset.y() * size_at_scale +
                                                         voxel_offset.z() * se::math::sq(size_at_scale)];
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const Eigen::Vector3i& voxel_coord,
                                         const int              scale,
                                         const VoxelData&       voxel_data) {
  int size_at_scale = this->size_li >> scale;
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_offset = voxel_offset / (1 << scale);
  block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                 voxel_offset.y() * size_at_scale +
                                 voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const Eigen::Vector3i& voxel_coord,
                                             const int              scale,
                                             const VoxelData&       voxel_data) {
  allocateDownTo(scale);
  int size_at_scale = this->size_li >> scale;
  Eigen::Vector3i voxel_offset = voxel_coord - this->coordinates_;
  voxel_offset = voxel_offset / (1 << scale);
  block_data_[VoxelBlock<T>::max_scale - scale][voxel_offset.x() +
                                                voxel_offset.y() * size_at_scale +
                                                voxel_offset.z() * se::math::sq(size_at_scale)] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const int voxel_idx) const {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size_li >> scale);
  }
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
    return init_data_;
  } else {
    return block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx];
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const int        voxel_idx,
                                         const VoxelData& voxel_data) {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size_li >> scale);
  }
  block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const int        voxel_idx,
                                             const VoxelData& voxel_data) {
  int remaining_voxel_idx = voxel_idx;
  int scale = 0;
  int size_at_scale_cu = this->size_cu;
  while (remaining_voxel_idx / size_at_scale_cu >= 1) {
    scale += 1;
    remaining_voxel_idx -= size_at_scale_cu;
    size_at_scale_cu = se::math::cu(this->size_li >> scale);
  }
  allocateDownTo(scale);
  block_data_[VoxelBlock<T>::max_scale - scale][remaining_voxel_idx] = voxel_data;
}

template <typename T>
inline typename VoxelBlock<T>::VoxelData
VoxelBlockSingle<T>::data(const int voxel_idx, const int scale) const {
  const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
  if (scale_idx < block_data_.size()) {
    return block_data_[scale_idx][voxel_idx];
  } else {
    return init_data_;
  }
}

template <typename T>
inline void VoxelBlockSingle<T>::setData(const int        voxel_idx,
                                         const int        scale,
                                         const VoxelData& voxel_data) {
  const size_t scale_idx = VoxelBlock<T>::max_scale - scale;
  block_data_[scale_idx][voxel_idx] = voxel_data;
}

template <typename T>
inline void VoxelBlockSingle<T>::setDataSafe(const int        voxel_idx,
                                             const int        scale,
                                             const VoxelData& voxel_data) {
  allocateDownTo(scale);
  setData(voxel_idx, scale, voxel_data);
}

template <typename T>
void VoxelBlockSingle<T>::allocateDownTo() {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) != 0) {
    for (int scale = VoxelBlock<T>::max_scale - block_data_.size(); scale >= 0; scale --) {
      int size_at_scale = this->size_li >> scale;
      int num_voxels_at_scale = se::math::cu(size_at_scale);
      VoxelData* voxel_data = new VoxelData[num_voxels_at_scale];
      initialiseData(voxel_data, num_voxels_at_scale);
      block_data_.push_back(voxel_data);
    }
    this->min_scale_ = 0;
  }
}

template <typename T>
void VoxelBlockSingle<T>::allocateDownTo(const int scale) {
  if (VoxelBlock<T>::max_scale - (block_data_.size() - 1) > static_cast<size_t>(scale)) {
    for (int scale_tmp = VoxelBlock<T>::max_scale - block_data_.size(); scale_tmp >= scale; scale_tmp --) {
      int size_at_scale_tmp = this->size_li >> scale_tmp;
      int num_voxels_at_scale_tmp = se::math::cu(size_at_scale_tmp);
      VoxelData* voxel_data = new VoxelData[num_voxels_at_scale_tmp];
      initialiseData(voxel_data, num_voxels_at_scale_tmp);
      block_data_.push_back(voxel_data);
    }
    this->min_scale_ = scale;
  }
}

template <typename T>
void VoxelBlockSingle<T>::deleteUpTo(const int scale) {
  if (this->min_scale_ == -1 || this->min_scale_ > scale) return;
  for (int scale_tmp = this->min_scale_; scale_tmp < scale; scale_tmp++) {
    auto data_at_scale = block_data_[this->max_scale - scale_tmp];
    delete[] data_at_scale;
    block_data_.pop_back();
  }
  this->min_scale_ = scale;
}

template <typename T>
void VoxelBlockSingle<T>::initFromBlock(const VoxelBlockSingle<T>& block) {
  this->code_          = block.code();
  this->size_          = block.size_;
  this->children_mask_ = block.children_mask();
  this->timestamp_     = block.timestamp();
  this->active_        = block.active();
  this->coordinates_   = block.coordinates();
  this->min_scale_     = block.min_scale();
  this->current_scale_ = block.current_scale();
  std::copy(block.childrenData(), block.childrenData() + 8, this->children_data_);
  if (block.min_scale() != -1) { // Verify that at least some mip-mapped level has been initialised.
    for (int scale = this->max_scale; scale >= block.min_scale(); scale--) {
      int size_at_scale = this->size_li >> scale;
      int num_voxels_at_scale = se::math::cu(size_at_scale);
      blockData().push_back(new typename T::VoxelData[num_voxels_at_scale]);
      std::copy(
          block.blockData()[VoxelBlock<T>::max_scale - scale],
          block.blockData()[VoxelBlock<T>::max_scale - scale] + num_voxels_at_scale,
          blockData()[VoxelBlock<T>::max_scale - scale]);
    }
  }
}

template <typename T>
void VoxelBlockSingle<T>::initialiseData(VoxelData* voxel_data, const int num_voxels) {
  for (int voxel_idx = 0; voxel_idx < num_voxels; voxel_idx++) {
    voxel_data[voxel_idx] = init_data_;
  }
}

} // namespace se

#endif // OCTREE_IMPL_HPP

