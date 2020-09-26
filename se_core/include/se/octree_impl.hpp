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

#ifndef OCTREE_IMPL_HPP
#define OCTREE_IMPL_HPP

namespace se {

template <typename T>
OctreeIterator<T> Octree<T>::begin() {
  return OctreeIterator<T>(this);
}

template <typename T>
OctreeIterator<T> Octree<T>::end() {
  return OctreeIterator<T>(nullptr);
}

template <typename T>
inline bool Octree<T>::contains(const int x, const int y, const int z) const {
  if (x >= 0 && x < size_ &&
      y >= 0 && y < size_ &&
      z >= 0 && z < size_) {
    return true;
  }
  return false;
}

template <typename T>
inline bool Octree<T>::contains(const Eigen::Vector3i& voxel_coord) const {
  return contains(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
}

template <typename T>
inline bool Octree<T>::contains(const Eigen::Vector3f& voxel_coord_f) const {
  if (voxel_coord_f.x() >= 0 && voxel_coord_f.x() < size_ &&
      voxel_coord_f.y() >= 0 && voxel_coord_f.y() < size_ &&
      voxel_coord_f.z() >= 0 && voxel_coord_f.z() < size_) {
    return true;
  }
  return false;
}

template <typename T>
bool Octree<T>::containsPoint(const Eigen::Vector3f& point_M) const {
  if (point_M.x() >= 0 && point_M.x() < dim_ &&
      point_M.y() >= 0 && point_M.y() < dim_ &&
      point_M.z() >= 0 && point_M.z() < dim_) {
    return true;
  }
  return false;
}



template <typename T>
inline int Octree<T>::get(const int  x,
                          const int  y,
                          const int  z,
                          VoxelData& data,
                          const int  min_scale) const {

  assert(min_scale < voxel_depth_);

  Node<T>* node = root_;
  if (!node) {
    data = T::initData();
    return size_; // size_ := map size
  }

  const unsigned min_node_size = std::max((1 << min_scale), (int) block_size);
  unsigned node_size = size_ >> 1; // size_ := map size
  // Initialize just to stop the compiler from complaining.
  int child_idx = -1;
  for (; node_size >= min_node_size; node_size = node_size >> 1) {
    child_idx  = ((x & node_size) > 0) + 2 * ((y & node_size) > 0) + 4 * ((z & node_size) > 0);
    Node<T>* node_tmp = node->child(child_idx);
    if (!node_tmp) {
      const int scale = se::math::log2_const(node->size() / 2);
      data = node->childData(child_idx);
      return scale;
    }
    node = node_tmp;
  }

  if (min_node_size == block_size) {
    const auto block = static_cast<VoxelBlockType *>(node);
    const int scale = std::max(min_scale, block->current_scale());
    data = block->data(Eigen::Vector3i(x, y, z), scale);
    return scale;
  } else {
    const int scale = se::math::log2_const(node->size() / 2);
    data = (node->parent())->childData(child_idx);
    return scale;
  }
}



template <typename T>
inline int Octree<T>::get(const Eigen::Vector3i& voxel_coord,
                          VoxelData&             data,
                          const int              scale) const {
  return get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data, scale);
}



template <typename T>
inline int Octree<T>::getAtPoint(const Eigen::Vector3f& point_M,
                                 VoxelData&             data,
                                 const int              scale) const {
  const Eigen::Vector3i voxel_coord = (inverse_voxel_dim_ * point_M).cast<int>();
  return get(voxel_coord, data, scale);
}



template <typename T>
inline void Octree<T>::set(const int        x,
                           const int        y,
                           const int        z,
                           const VoxelData& data) {
  Node<T>* node = root_;
  if (!node) {
    return;
  }

  unsigned node_size = size_ >> 1;
  for (; node_size >= block_size; node_size = node_size >> 1) {
    Node<T>* node_tmp = node->child(
        (x & node_size) > 0,
        (y & node_size) > 0,
        (z & node_size) > 0);
    if (!node_tmp) {
      return;
    }
    node = node_tmp;
  }

  static_cast<VoxelBlockType *>(node)->setData(Eigen::Vector3i(x, y, z), data);
}



template <typename T>
inline void Octree<T>::set(const Eigen::Vector3i& voxel_coord,
                           const VoxelData&       data) {
  return set(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), data);
}



template <typename T>
inline void Octree<T>::setAtPoint(const Eigen::Vector3f& point_M,
                                  const VoxelData&       data) {
  const Eigen::Vector3i voxel_coord = (inverse_voxel_dim_ * point_M).cast<int>();
  return set(voxel_coord, data);
}



template <typename T>
inline Eigen::Vector3f Octree<T>::voxelToPoint(const Eigen::Vector3i& voxel_coord) const {
  return voxelToPoint(voxel_coord.cast<float>());
}



template <typename T>
inline Eigen::Vector3f Octree<T>::voxelToPoint(const Eigen::Vector3f& voxel_coord_f) const {
  return voxel_coord_f * voxel_dim_;
}



template <typename T>
inline Eigen::Vector3i Octree<T>::pointToVoxel(const Eigen::Vector3f& point_M) const {
  // For some reason C++ doesn't like the cast being right after the function
  // call.
  const Eigen::Vector3f voxel_coord = pointToVoxelF(point_M);
  return voxel_coord.cast<int>();
}



template <typename T>
inline Eigen::Vector3f Octree<T>::pointToVoxelF(const Eigen::Vector3f& point_M) const {
  return point_M * inverse_voxel_dim_;
}



template <typename T>
template <bool safe>
inline std::array<typename Octree<T>::VoxelData, 6> Octree<T>::getFaceNeighbours(
    const int x, const int y, const int z) const {

  std::array<typename Octree<T>::VoxelData, 6> neighbor_data;

  for (size_t i = 0; i < 6; ++i) {
    // Compute the neighbor voxel coordinates.
    const int neighbor_x = x + face_neighbor_offsets[i].x();
    const int neighbor_y = y + face_neighbor_offsets[i].y();
    const int neighbor_z = z + face_neighbor_offsets[i].z();

    if(safe) {
      if(    (neighbor_x >= 0) and (neighbor_x < size())
         and (neighbor_y >= 0) and (neighbor_y < size())
         and (neighbor_z >= 0) and (neighbor_z < size())) {
        // The neighbor voxel is inside the map, get its value.
        auto& neighbor_data_i = neighbor_data[i];
        get(neighbor_x, neighbor_y, neighbor_z, neighbor_data_i);
      } else {
        // The neighbor voxel is outside the map, set the value to empty.
        neighbor_data[i] = T::invalid();
      }
    } else {
      // Get the value of the neighbor voxel.
      auto& neighbor_data_i = neighbor_data[i];
      get(neighbor_x, neighbor_y, neighbor_z, neighbor_data_i);
    }
  }

  return neighbor_data;
}



template <typename T>
inline int Octree<T>::get(const int       x,
                          const int       y,
                          const int       z,
                          VoxelBlockType* cached_block,
                          VoxelData&      data,
                          const int       min_scale) const {

  if(cached_block != NULL) {
    const Eigen::Vector3i voxel_coord = Eigen::Vector3i(x, y, z);
    const Eigen::Vector3i lower_coord = cached_block->coordinates();
    const Eigen::Vector3i upper_coord = lower_coord + Eigen::Vector3i::Constant(block_size - 1);
    const int contained =
      ((voxel_coord.array() >= lower_coord.array()) && (voxel_coord.array() <= upper_coord.array())).all();
    if(contained){
      const int scale = std::max(cached_block->current_scale(), min_scale);
      data = cached_block->data(Eigen::Vector3i(x, y, z), scale);
      return scale;
    }
  }

  return get(x, y, z, data, min_scale);
}



template <typename T>
inline int Octree<T>::get(const Eigen::Vector3i& voxel_coord,
                          VoxelBlockType*        cached_block,
                          VoxelData&             data,
                          const int              min_scale) const {

  return get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), cached_block, data, min_scale);
}



template <typename T>
inline int Octree<T>::getAtPoint(const Eigen::Vector3f& point_M,
                                 VoxelBlockType*        cached_block,
                                 VoxelData&             data,
                                 const int              min_scale) const {

  const Eigen::Vector3i voxel_coord =
      (point_M.homogeneous() * Eigen::Vector4f::Constant(inverse_voxel_dim_)).template head<3>().template cast<int>();

  return get(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), cached_block, data, min_scale);
}



template <typename T>
void Octree<T>::deleteNode(Node<T> **node){

  if(*node){
    for (int child_idx = 0; child_idx < 8; child_idx++) {
      if((*node)->child(child_idx)){
        deleteNode(&(*node)->child(child_idx));
      }
    }
    if(!(*node)->isBlock()){
      delete *node;
      *node = NULL;
    }
  }
}



template <typename T>
void Octree<T>::init(int size, float dim) {
  size_ = size;
  dim_ = dim;
  voxel_dim_ = dim_ / size_;
  inverse_voxel_dim_ = size_ / dim_;
  voxel_depth_ = log2(size);
  num_levels_ = voxel_depth_ + 1;
  max_block_scale_ = log2(block_size);
  block_depth_ = voxel_depth_ - max_block_scale_;
  root_ = pool_.root();
  root_->size(size);
  reserved_ = 1024;
  keys_at_depth_.resize(reserved_, 0);
}



template <typename T>
inline typename Octree<T>::VoxelBlockType* Octree<T>::fetch(const int x, const int y,
   const int z) const {

  Node<T>* node = root_;
  if(!node) {
    return nullptr;
  }

  // Get the block.
  unsigned node_size = size_ / 2;
  for(; node_size >= block_size; node_size /= 2){
    node = node->child((x & node_size) > 0u, (y & node_size) > 0u, (z & node_size) > 0u);
    if(!node){
      return nullptr;
    }
  }
  return static_cast<VoxelBlockType* > (node);
}



template <typename T>
inline typename Octree<T>::VoxelBlockType* Octree<T>::fetch(const Eigen::Vector3i& voxel_coord) const {
  return fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
}



template <typename T>
inline Node<T>* Octree<T>::fetchNode(const int x, const int y, const int z, const int depth) const {

  Node<T>* node = root_;
  if(!node) {
    return nullptr;
  }

  // Get the block.
  unsigned node_size = size_ / 2;
  for(int d = 1; node_size >= block_size && d <= depth; node_size /= 2, ++d){
    node = node->child((x & node_size) > 0u, (y & node_size) > 0u, (z & node_size) > 0u);
    if(!node){
      return nullptr;
    }
  }
  return node;
}



template <typename T>
inline Node<T>* Octree<T>::fetchNode(const Eigen::Vector3i& voxel_coord,
                                     const int              depth) const {
  return fetchNode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), depth);
}



template <typename T>
Node<T>* Octree<T>::insert(const int x,
                           const int y,
                           const int z,
                           const int depth,
                           Node<T>*  init_octant) {

  // Make sure we have enough space on buffers
  if(depth >= block_depth_) {
    pool_.reserveNodes(block_depth_);
    pool_.reserveBlocks(1);
  } else {
    pool_.reserveNodes(depth);
  }

  Node<T>* node = root_;
  // Should not happen if octree has been initialised properly
  if(!node) {
    root_ = pool_.root();
    root_->code(0);
    root_->size(size_);
    node = root_;
  }

  key_t key = keyops::encode(x, y, z, depth, voxel_depth_);
  const unsigned int shift = MAX_BITS - voxel_depth_ - 1;

  unsigned node_size = size_ / 2;
  for(int d = 1; node_size >= block_size && d <= depth; node_size /= 2, ++d){
    const int child_idx = ((x & node_size) > 0) +  2 * ((y & node_size) > 0)
      +  4 * ((z & node_size) > 0);

    // std::cout << "Level: " << d << std::endl;
    Node<T>* node_tmp = node->child(child_idx);
    if(!node_tmp){
      const key_t prefix = keyops::code(key) & MASK[d + shift];
      if(node_size == block_size) {
        if (init_octant == nullptr) {
          node_tmp = pool_.acquireBlock();
          static_cast<VoxelBlockType *>(node_tmp)->active(true);
        } else {
          node_tmp = pool_.acquireBlock(static_cast<VoxelBlockType *>(init_octant));
        }
        static_cast<VoxelBlockType *>(node_tmp)->coordinates(
            Eigen::Vector3i(unpack_morton(prefix)));
        static_cast<VoxelBlockType *>(node_tmp)->code(prefix | d);
        node_tmp->parent() = node;
        node->children_mask(node->children_mask() | (1 << child_idx));
      } else {
        if (init_octant == nullptr) {
          node_tmp = pool_.acquireNode();
          node_tmp->size(node_size);
        } else {
          node_tmp = pool_.acquireNode(init_octant);
        }
        node_tmp->parent() = node;
        node_tmp->code(prefix | d);
        node->children_mask(node->children_mask() | (1 << child_idx));
        // std::cout << "coords: "
        //   << keyops::decode(keyops::code(node_tmp->code())) << std::endl;
      }
      node->child(child_idx) = node_tmp;
    }
    node = node_tmp;
  }
  return node;
}



template <typename T>
typename Octree<T>::VoxelBlockType* Octree<T>::insert(const int       x,
                                                      const int       y,
                                                      const int       z,
                                                      VoxelBlockType* init_block) {
  return static_cast<VoxelBlockType* >(insert(x, y, z, voxel_depth_, init_block));
}



template <typename T>
template <typename ValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        ValueSelector          select_value,
                                        const int              min_scale) const {
  return interp(voxel_coord_f, select_value, select_value, min_scale);
}



template <typename T>
template <typename ValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        ValueSelector          select_value,
                                        const int              min_scale,
                                        bool&                  is_valid) const {
  return interp(voxel_coord_f, select_value, select_value, min_scale, is_valid);
}



template <typename T>
template <typename NodeValueSelector,
          typename VoxelValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        NodeValueSelector      select_node_value,
                                        VoxelValueSelector     select_voxel_value,
                                        const int              min_scale) const {

  // The return type of the select_value() function. Since it can be a lambda
  // function, an argument needs to be passed to it before deducing the return
  // type.
  typedef decltype(select_voxel_value(T::initData())) value_t;

  int iter = 0;
  int target_scale = min_scale;
  value_t voxel_values[8] = { select_voxel_value(T::initData()) };
  Eigen::Vector3f factor;
  while (iter < 3) {
    const int stride = 1 << target_scale;
    const Eigen::Vector3f scaled_voxel_coord_f = 1.f / stride * voxel_coord_f - sample_offset_frac_;
    factor = math::fracf(scaled_voxel_coord_f);
    const Eigen::Vector3i base_coord = stride * scaled_voxel_coord_f.cast<int>();
    if ((base_coord.array() < 0).any() ||
        ((base_coord + Eigen::Vector3i::Constant(stride)).array() >= size_).any()) {
      return {select_voxel_value(T::initData()), target_scale};
    }
    int interp_scale = internal::gather_values(*this, base_coord, target_scale,
        select_node_value, select_voxel_value, voxel_values);
    if (interp_scale == target_scale) {
      break;
    } else {
      target_scale = interp_scale;
    }
    iter++;
  }

  // Interpolate the value based on the fractional part.
  return {(((voxel_values[0] * (1 - factor.x())
          + voxel_values[1] * factor.x()) * (1 - factor.y())
          + (voxel_values[2] * (1 - factor.x())
          + voxel_values[3] * factor.x()) * factor.y())
          * (1 - factor.z())
          + ((voxel_values[4] * (1 - factor.x())
          + voxel_values[5] * factor.x())
          * (1 - factor.y())
          + (voxel_values[6] * (1 - factor.x())
          + voxel_values[7] * factor.x())
          * factor.y()) * factor.z()), target_scale};
}



template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        NodeValueSelector      select_node_value,
                                        VoxelValueSelector     select_voxel_value,
                                        const int              min_scale,
                                        bool&                  is_valid) const {

  auto select_weight = [](const auto& data) { return data.y; };

  // The return types of the select() and select_weight() functions. Since they
  // can be lambda functions, an argument needs to be passed to the, before
  // deducing the return type.
  typedef decltype(select_voxel_value(T::initData())) value_t;
  typedef decltype(select_weight(T::initData())) weight_t;

  int iter = 0;
  int target_scale = min_scale;
  value_t voxel_values[8] = { select_voxel_value(T::initData()) };
  weight_t voxel_weights[8];
  Eigen::Vector3f factor;
  while (iter < 3) {
    const int stride = 1 << target_scale;
    const Eigen::Vector3f scaled_voxel_coord_f = 1.f / stride * voxel_coord_f - sample_offset_frac_;
    factor =  math::fracf(scaled_voxel_coord_f);
    const Eigen::Vector3i base_coord = stride * scaled_voxel_coord_f.cast<int>();
    if ((base_coord.array() < 0).any() ||
        ((base_coord + Eigen::Vector3i::Constant(stride)).array() >= size_).any()) {
      is_valid = false;
      return {select_voxel_value(T::initData()), target_scale};
    }

    int interp_scale = se::internal::gather_values(*this, base_coord, target_scale, select_node_value, select_voxel_value, voxel_values);
    se::internal::gather_values(
        *this, base_coord, target_scale, select_weight, select_weight, voxel_weights);

    if (interp_scale == target_scale) {
      break;
    } else {
      target_scale = interp_scale;
    }
    iter++;
  }

  for (int i = 0; i < 8; ++i) {
    if (voxel_weights[i] == 0) {
      is_valid = false;
      return {select_voxel_value(T::initData()), -1};
    }
  }

  is_valid = true;
  return {(((voxel_values[0] * (1 - factor.x())
          + voxel_values[1] * factor.x()) * (1 - factor.y())
          + (voxel_values[2] * (1 - factor.x())
          + voxel_values[3] * factor.x()) * factor.y())
          * (1 - factor.z())
          + ((voxel_values[4] * (1 - factor.x())
          + voxel_values[5] * factor.x())
          * (1 - factor.y())
          + (voxel_values[6] * (1 - factor.x())
          + voxel_values[7] * factor.x())
          * factor.y()) * factor.z()), target_scale};
}



template <typename T>
template <typename ValueSelector>
inline std::pair<float, int> Octree<T>::interpAtPoint(
    const Eigen::Vector3f& point_M,
    ValueSelector          select_value,
    const int              min_scale) const {
  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return interp(voxel_coord_f, select_value, min_scale);
}



template <typename T>
template <typename ValueSelector>
inline std::pair<float, int> Octree<T>::interpAtPoint(
    const Eigen::Vector3f& point_M,
    ValueSelector          select_value,
    const int              min_scale,
    bool&                  is_valid) const {
  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return interp(voxel_coord_f, select_value, min_scale, is_valid);
}



template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
inline std::pair<float, int> Octree<T>::interpAtPoint(
    const Eigen::Vector3f& point_M,
    NodeValueSelector      select_node_value,
    VoxelValueSelector     select_voxel_value,
    const int              min_scale) const {
  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return interp(voxel_coord_f, select_node_value, select_voxel_value, min_scale);
}



template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
inline std::pair<float, int> Octree<T>::interpAtPoint(
    const Eigen::Vector3f& point_M,
    NodeValueSelector      select_node_value,
    VoxelValueSelector     select_voxel_value,
    const int              min_scale,
    bool&                  is_valid) const {
  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return interp(voxel_coord_f, select_node_value, select_voxel_value, min_scale, is_valid);
}



template <typename T>
template <typename ValueSelector>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f,
                                ValueSelector          select_value,
                                const int              min_scale) const {

  auto get_value = [&](int x, int y, int z, VoxelBlockType* block) {
    VoxelData data;
    get(x, y, z, block, data, min_scale);
    select_value(data);
  };

  return gradImpl(voxel_coord_f, get_value, min_scale);
}

template <typename T>
template <typename ValueSelector, typename ValidChecker>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f,
                                ValueSelector          select_value,
                                ValidChecker           check_is_valid,
                                bool&                  is_valid,
                                const int              min_scale) const {

  is_valid = true;

  auto get_values = [&](int x, int y, int z, VoxelBlockType* block) {
    VoxelData data;
    get(x, y, z, block, data, min_scale);
    if (is_valid) {
      is_valid = check_is_valid(data);
    }
    return select_value(data);
  };

  return gradImpl(voxel_coord_f, get_values, min_scale);
}

template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f,
                                NodeValueSelector      select_node_value,
                                VoxelValueSelector     select_voxel_value,
                                const int              min_scale) const {

  auto get_values = [&](int x, int y, int z, VoxelBlockType* block) {
    VoxelData data;
    const unsigned int scale = get(x, y, z, block, data, min_scale);
    if (scale > VoxelBlockType::max_scale) {
      return select_node_value(data);
    } else {
      return select_voxel_value(data);
    }
  };

  return gradImpl(voxel_coord_f, get_values, min_scale);
}

template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector, typename ValidChecker>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f,
                                NodeValueSelector      select_node_value,
                                VoxelValueSelector     select_voxel_value,
                                ValidChecker           check_is_valid,
                                bool&                  is_valid,
                                const int              min_scale) const {

  is_valid = true;

  auto get_values = [&](int x, int y, int z, VoxelBlockType* block) {
    VoxelData data;
    const unsigned int scale = get(x, y, z, block, data, scale);
    if (is_valid) {
      is_valid = check_is_valid(data);
    }
    if (scale > VoxelBlockType::max_scale_) {
      return select_node_value(data);
    } else {
      return select_voxel_value(data);
    }
  };

  return gradImpl(voxel_coord_f, get_values, min_scale);
}



template <typename T>
template <typename ValueSelector>
inline Eigen::Vector3f Octree<T>::gradAtPoint(const Eigen::Vector3f& point_M,
                                              ValueSelector          select_value,
                                              const int              min_scale) const {

  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return grad(voxel_coord_f, select_value, min_scale);
}



template <typename T>
template <typename ValueSelector, typename ValidChecker>
inline Eigen::Vector3f Octree<T>::gradAtPoint(const Eigen::Vector3f& point_M,
                                              ValueSelector          select_value,
                                              ValidChecker           check_is_valid,
                                              bool&                  is_valid,
                                              const int              min_scale) const {

  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return grad(voxel_coord_f, select_value, check_is_valid, is_valid, min_scale);
}



template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
inline Eigen::Vector3f Octree<T>::gradAtPoint(const Eigen::Vector3f& point_M,
                                              NodeValueSelector      select_node_value,
                                              VoxelValueSelector     select_voxel_value,
                                              const int              min_scale) const {

  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return grad(voxel_coord_f, select_node_value, select_voxel_value, min_scale);
}



template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector, typename ValidChecker>
inline Eigen::Vector3f Octree<T>::gradAtPoint(const Eigen::Vector3f& point_M,
                                              NodeValueSelector      select_node_value,
                                              VoxelValueSelector     select_voxel_value,
                                              ValidChecker           check_is_valid,
                                              bool&                  is_valid,
                                              const int              min_scale) const {

  const Eigen::Vector3f voxel_coord_f = inverse_voxel_dim_ * point_M;
  return grad(voxel_coord_f, select_node_value, select_voxel_value, check_is_valid, is_valid, min_scale);
}



template <typename T>
template <typename ValuesGetter>
Eigen::Vector3f Octree<T>::gradImpl(const Eigen::Vector3f& voxel_coord_f,
                                    ValuesGetter           get_values,
                                    const int              min_scale) const {

  int iter = 0;
  int scale = min_scale;
  int last_scale = scale;
  Eigen::Vector3f factor = Eigen::Vector3f::Constant(0);
  Eigen::Vector3f gradient = Eigen::Vector3f::Constant(0);
  while (iter < 3) {
    const int stride = 1 << scale;
    const Eigen::Vector3f scaled_voxel_coord_f = 1.f / stride * voxel_coord_f - sample_offset_frac_;
    factor =  math::fracf(scaled_voxel_coord_f);
    const Eigen::Vector3i base_coord = stride * scaled_voxel_coord_f.cast<int>();
    Eigen::Vector3i lower_lower_coord = (base_coord - stride * Eigen::Vector3i::Constant(1)).cwiseMax(Eigen::Vector3i::Constant(0));
    Eigen::Vector3i lower_upper_coord = base_coord.cwiseMax(Eigen::Vector3i::Constant(0));
    Eigen::Vector3i upper_lower_coord = (base_coord + stride * Eigen::Vector3i::Constant(1)).cwiseMin(
        Eigen::Vector3i::Constant(size_) - Eigen::Vector3i::Constant(1));
    Eigen::Vector3i upper_upper_coord = (base_coord + stride * Eigen::Vector3i::Constant(2)).cwiseMin(
        Eigen::Vector3i::Constant(size_) - Eigen::Vector3i::Constant(1));
    Eigen::Vector3i & lower_coord = lower_upper_coord;
    Eigen::Vector3i & upper_coord = upper_lower_coord;

    VoxelBlockType* block = fetch(base_coord.x(), base_coord.y(), base_coord.z());

    gradient.x() = (((get_values(upper_lower_coord.x(), lower_coord.y(), lower_coord.z(), block)
                    - get_values(lower_lower_coord.x(), lower_coord.y(), lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_upper_coord.x(), lower_coord.y(), lower_coord.z(), block)
                    - get_values(lower_upper_coord.x(), lower_coord.y(), lower_coord.z(), block)) * factor.x()) * (1 - factor.y())
                  + ((get_values(upper_lower_coord.x(), upper_coord.y(), lower_coord.z(), block)
                    - get_values(lower_lower_coord.x(), upper_coord.y(), lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_upper_coord.x(), upper_coord.y(), lower_coord.z(), block)
                    - get_values(lower_upper_coord.x(), upper_coord.y(), lower_coord.z(), block)) * factor.x()) * factor.y()) * (1 - factor.z())
                 + (((get_values(upper_lower_coord.x(), lower_coord.y(), upper_coord.z(), block)
                    - get_values(lower_lower_coord.x(), lower_coord.y(), upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_upper_coord.x(), lower_coord.y(), upper_coord.z(), block)
                    - get_values(lower_upper_coord.x(), lower_coord.y(), upper_coord.z(), block)) * factor.x()) * (1 - factor.y())
                  + ((get_values(upper_lower_coord.x(), upper_coord.y(), upper_coord.z(), block)
                    - get_values(lower_lower_coord.x(), upper_coord.y(), upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_upper_coord.x(), upper_coord.y(), upper_coord.z(), block)
                    - get_values(lower_upper_coord.x(), upper_coord.y(), upper_coord.z(), block)) * factor.x()) * factor.y()) * factor.z();
    if (scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }

    gradient.y() = (((get_values(lower_coord.x(), upper_lower_coord.y(), lower_coord.z(), block)
                    - get_values(lower_coord.x(), lower_lower_coord.y(), lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_lower_coord.y(), lower_coord.z(), block)
                    - get_values(upper_coord.x(), lower_lower_coord.y(), lower_coord.z(), block)) * factor.x()) * (1 - factor.y())
                  + ((get_values(lower_coord.x(), upper_upper_coord.y(), lower_coord.z(), block)
                    - get_values(lower_coord.x(), lower_upper_coord.y(), lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_upper_coord.y(), lower_coord.z(), block)
                    - get_values(upper_coord.x(), lower_upper_coord.y(), lower_coord.z(), block)) * factor.x()) * factor.y()) * (1 - factor.z())
                 + (((get_values(lower_coord.x(), upper_lower_coord.y(), upper_coord.z(), block)
                    - get_values(lower_coord.x(), lower_lower_coord.y(), upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_lower_coord.y(), upper_coord.z(), block)
                    - get_values(upper_coord.x(), lower_lower_coord.y(), upper_coord.z(), block)) * factor.x()) * (1 - factor.y())
                  + ((get_values(lower_coord.x(), upper_upper_coord.y(), upper_coord.z(), block)
                    - get_values(lower_coord.x(), lower_upper_coord.y(), upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_upper_coord.y(), upper_coord.z(), block)
                    - get_values(upper_coord.x(), lower_upper_coord.y(), upper_coord.z(), block)) * factor.x()) * factor.y()) * factor.z();
    if (scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }

    gradient.z() = (((get_values(lower_coord.x(), lower_coord.y(), upper_lower_coord.z(), block)
                    - get_values(lower_coord.x(), lower_coord.y(), lower_lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), lower_coord.y(), upper_lower_coord.z(), block)
                    - get_values(upper_coord.x(), lower_coord.y(), lower_lower_coord.z(), block)) * factor.x()) * (1 - factor.y())
                  + ((get_values(lower_coord.x(), upper_coord.y(), upper_lower_coord.z(), block)
                    - get_values(lower_coord.x(), upper_coord.y(), lower_lower_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_coord.y(), upper_lower_coord.z(), block)
                    - get_values(upper_coord.x(), upper_coord.y(), lower_lower_coord.z(), block)) * factor.x()) * factor.y()) * (1 - factor.z())
                 + (((get_values(lower_coord.x(), lower_coord.y(), upper_upper_coord.z(), block)
                    - get_values(lower_coord.x(), lower_coord.y(), lower_upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), lower_coord.y(), upper_upper_coord.z(), block)
                    - get_values(upper_coord.x(), lower_coord.y(), lower_upper_coord.z(), block)) * factor.x()) * (1 - factor.y())
                   +((get_values(lower_coord.x(), upper_coord.y(), upper_upper_coord.z(), block)
                    - get_values(lower_coord.x(), upper_coord.y(), lower_upper_coord.z(), block)) * (1 - factor.x())
                    +(get_values(upper_coord.x(), upper_coord.y(), upper_upper_coord.z(), block)
                    - get_values(upper_coord.x(), upper_coord.y(), lower_upper_coord.z(), block)) * factor.x()) * factor.y()) * factor.z();
    if (scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }
    break;
  }

  return (0.5f * voxel_dim_) * gradient;
}



template <typename T>
int Octree<T>::blockCount(){
  return pool_.blockBufferSize();
}



template <typename T>
int Octree<T>::blockCountRecursive(Node<T>* node){

  if(!node) return 0;

  if(node->isBlock()){
    return 1;
  }

  int sum = 0;

  for (int child_idx = 0; child_idx < 8; child_idx++){
    sum += blockCountRecursive(node->child(child_idx));
  }

  return sum;
}



template <typename T>
int Octree<T>::nodeCount(){
  return pool_.nodeBufferSize();
}



template <typename T>
int Octree<T>::nodeCountRecursive(Node<T>* node){
  if (!node) {
    return 0;
  }

  int node_count = 1;
  for (int child_idx = 0; child_idx < 8; ++child_idx) {
    node_count += (node_count ? nodeCountRecursive((node)->child(child_idx)) : 0);
  }
  return node_count;
}



template <typename T>
void Octree<T>::reserveBuffers(const int num_blocks){

  if(num_blocks > reserved_){
    // std::cout << "Reserving " << n << " entries in allocation buffers" << std::endl;
    keys_at_depth_.resize(num_blocks);
    reserved_ = num_blocks;
  }
  pool_.reserveBlocks(num_blocks);
}



template <typename T>
bool Octree<T>::allocate(key_t* keys, int num_elem){

#if defined(_OPENMP) && !defined(__clang__)
  __gnu_parallel::sort(keys, keys+num_elem);
#else
std::sort(keys, keys + num_elem);
#endif

  num_elem = algorithms::filter_ancestors(keys, num_elem, voxel_depth_);
  reserveBuffers(num_elem); // Reserve memory for blocks

  int last_elem = 0;
  bool success = false;

  const unsigned int shift = MAX_BITS - voxel_depth_ - 1;
  for (int depth = 1; depth <= block_depth_; depth++){
    const key_t mask = MASK[depth + shift] | SCALE_MASK;
    compute_prefix(keys, keys_at_depth_.data(), num_elem, mask);
    last_elem = algorithms::unique_multiscale(keys_at_depth_.data(), num_elem);
    success = allocate_depth(keys_at_depth_.data(), last_elem, depth);
  }
  return success;
}



template <typename T>
bool Octree<T>::allocate_depth(key_t* octant_keys, int num_tasks, int target_depth){

  pool_.reserveNodes(num_tasks); // Reserve memory for nodes

#pragma omp parallel for
  for (int i = 0; i < num_tasks; i++){
    Node<T>** node = &root_;
    const key_t octant_key = keyops::code(octant_keys[i]);
    const int octant_depth = keyops::depth(octant_keys[i]);
    if (octant_depth < target_depth) continue;

    int octant_size = size_ / 2;
    for (int depth = 1; depth <= target_depth; ++depth){
      const int child_idx = se::child_idx(octant_key, depth, voxel_depth_);
      Node<T>* parent = *node;
      node = &(*node)->child(child_idx);

      if (!(*node)) {
        if (depth == block_depth_) {
          *node = pool_.acquireBlock();
          (*node)->parent() = parent;
          (*node)->size(octant_size);
          static_cast<VoxelBlockType *>(*node)->coordinates(Eigen::Vector3i(unpack_morton(octant_key)));
          static_cast<VoxelBlockType *>(*node)->active(true);
          static_cast<VoxelBlockType *>(*node)->code(octant_key | depth);
          parent->children_mask(parent->children_mask() | (1 << child_idx));
        } else {
          *node = pool_.acquireNode();
          (*node)->parent() = parent;
          (*node)->code(octant_key | depth);
          (*node)->size(octant_size);
          parent->children_mask(parent->children_mask() | (1 << child_idx));
        }
      }
      octant_size /= 2;
    }
  }
  return true;
}



template <typename T>
void Octree<T>::getBlockList(std::vector<VoxelBlockType*>& block_list, bool active){
  Node<T>* node = root_;
  if(!node) return;
  if(active) getActiveBlockList(node, block_list);
  else getAllocatedBlockList(node, block_list);
}



template <typename T>
void Octree<T>::getActiveBlockList(Node<T>* node,
    std::vector<VoxelBlockType*>& block_list){
  if(!node) return;
  std::queue<Node<T> *> node_queue;
  node_queue.push(node);
  while(!node_queue.empty()){
    Node<T>* node_tmp = node_queue.front();
    node_queue.pop();

    if(node_tmp->isBlock()){
      VoxelBlockType* block = static_cast<VoxelBlockType *>(node_tmp);
      if(block->active()) block_list.push_back(block);
      continue;
    }

    for(int child_idx = 0; child_idx < 8; ++child_idx){
      if(node_tmp->child(child_idx)) node_queue.push(node_tmp->child(child_idx));
    }
  }
}



template <typename T>
void Octree<T>::getAllocatedBlockList(Node<T>* ,
    std::vector<VoxelBlockType*>& block_list){
  auto& block_buffer = pool_.blockBuffer();
  for(unsigned int i = 0; i < block_buffer.size(); ++i) {
    block_list.push_back(block_buffer[i]);
  }
}



template <typename T>
void Octree<T>::save(const std::string& filename) {
  std::ofstream os (filename, std::ios::binary);
  os.write(reinterpret_cast<char *>(&size_), sizeof(size_));
  os.write(reinterpret_cast<char *>(&dim_), sizeof(dim_));

  auto& node_buffer = pool_.nodeBuffer();
  size_t num_nodes = node_buffer.size();
  os.write(reinterpret_cast<char *>(&num_nodes), sizeof(size_t));
  for(size_t i = 0; i < num_nodes; ++i)
    internal::serialise(os, *node_buffer[i]);

  auto& block_buffer = pool_.blockBuffer();
  size_t num_blocks = block_buffer.size();
  os.write(reinterpret_cast<char *>(&num_blocks), sizeof(size_t));
  for(size_t i = 0; i < num_blocks; ++i)
    internal::serialise(os, *block_buffer[i]);
}



template <typename T>
void Octree<T>::load(const std::string& filename) {
  std::cout << "Loading octree from disk... " << filename << std::endl;
  std::ifstream is (filename, std::ios::binary);
  int size;
  float dim;

  is.read(reinterpret_cast<char *>(&size), sizeof(size));
  is.read(reinterpret_cast<char *>(&dim), sizeof(dim));

  init(size, dim);

  size_t num_nodes = 0;
  is.read(reinterpret_cast<char *>(&num_nodes), sizeof(size_t));
  pool_.reserveNodes(num_nodes);
  std::cout << "Reading " << num_nodes << " nodes " << std::endl;
  for(size_t i = 0; i < num_nodes; ++i) {
    // Node      := Temporary block on the stack that's only used to read block information from the file and to
    //              initialise the inserted node.
    Node<T> node;
    internal::deserialise(node, is);
    Eigen::Vector3i node_coord = keyops::decode(node.code());
    insert(node_coord.x(), node_coord.y(), node_coord.z(), keyops::depth(node.code()), &node);
  }

  size_t num_blocks = 0;
  is.read(reinterpret_cast<char *>(&num_blocks), sizeof(size_t));
  std::cout << "Reading " << num_blocks << " blocks " << std::endl;
  for(size_t i = 0; i < num_blocks; ++i) {
    // block     := Temporary block on the stack that's only used to read block information from the file and to
    //              initialise the inserted block.
    VoxelBlockType block;
    internal::deserialise(block, is);
    Eigen::Vector3i block_coord = block.coordinates();
    insert(block_coord.x(), block_coord.y(), block_coord.z(), keyops::depth(block.code()), &block);
  }
}

}

template <typename FieldType>
const Eigen::Vector3f se::Octree<FieldType>::sample_offset_frac_ =
    Eigen::Vector3f::Constant(SAMPLE_POINT_POSITION);

#endif // OCTREE_IMPL_HPP

