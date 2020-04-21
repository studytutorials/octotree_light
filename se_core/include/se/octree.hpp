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

#ifndef OCTREE_H
#define OCTREE_H

#include <cstring>
#include <algorithm>
#include "utils/math_utils.h"
#include "octree_defines.h"
#include "utils/morton_utils.hpp"
#include "octant_ops.hpp"

#if defined(_OPENMP) && !defined(__clang__)
#include <parallel/algorithm>
#endif

#include <array>
#include <tuple>
#include <queue>
#include <unordered_set>
#include "node.hpp"
#include "utils/memory_pool.hpp"
#include "algorithms/unique.hpp"
#include "geometry/aabb_collision.hpp"
#include "interpolation/interp_gather.hpp"
#include "se/interpolation/interp_gather.hpp"
#include "neighbors/neighbor_gather.hpp"

namespace se {

/*
 * Value between 0.f and 1.f. Defines the sample point position relative to the
 * voxel anchor.  E.g. 0.5f means that the point sample corresponds to the
 * centre of the voxel.
 */
#define SAMPLE_POINT_POSITION 0.5f

template <typename T>
class VoxelBlockRayIterator;

template <typename T>
class node_iterator;

/*! \brief The main octree class.
 * Its non-leaf nodes are of type Node and its leaf nodes of type VoxelBlock.
 * For a minimal working example of the kind of struct needed as a template
 * parameter see ExampleVoxelT.
 */
template <typename T>
class Octree
{

public:
  typedef typename T::VoxelData VoxelData;

  // Compile-time constant expressions
  // # of voxels per side in a voxel block
  static constexpr unsigned int block_size = BLOCK_SIZE;
  // maximum tree depth in bits
  static constexpr unsigned int max_voxel_depth = ((sizeof(key_t) * 8) / 3);
  // Tree depth at which blocks are found
  static constexpr unsigned int max_block_depth = max_voxel_depth - math::log2_const(BLOCK_SIZE);

  static const Eigen::Vector3f sample_offset_frac_;


  Octree(){
  };

  ~Octree(){
  }

  /*! \brief Initialises the octree attributes
   * \param size number of voxels per side of the cube
   * \param dim cube extension per side, in meter
   */
  void init(int size, float dim);

  inline int size() const { return size_; }
  inline float dim() const { return dim_; }
  inline int numLevels() const { return num_levels_; }
  inline int voxelDepth() const { return voxel_depth_; }
  inline int maxBlockScale() const { return max_block_scale_; }
  inline int blockDepth() const { return block_depth_; }
  inline Node<T>* const root() const { return root_; }

  /*! \brief Sets voxel data at coordinates (x,y,z), if not present it
   * allocates it. This method is not thread safe.
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  void set(const int x, const int y, const int z, const VoxelData data);

  /*! \brief Retrieves voxel value at coordinates (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  VoxelData get(const int x, const int y, const int z) const;
  VoxelData get_fine(const int x, const int y, const int z, const int scale = 0) const;

  /*! \brief Retrieves voxel values for the neighbors of voxel at coordinates
   * (x,y,z)
   * If the safe template variable is true, then proper checks will be used so
   * that neighboring voxels outside the map will have a value of empty at a
   * cost of performance. Otherwise if the safe template variable is false,
   * neighboring voxels outside the map will not be detected and will have the
   * value of some voxel inside the map. If you are certain your code will not
   * ask for the neighbors of voxels at the edge of the map, then the non-safe
   * version should be safe to use and will result in better performance.
   *
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   * \return An std::array with the values of the 6 neighboring voxels. The
   * voxels are returned in the order: -z -y -x +x +y +z. Neighboring voxels
   * that are not allocated have the initial value. Neighboring voxels that are
   * outside the map have the empty value if safe is true, otherwise their
   * value is undetermined.
   *
   * \todo The implementation is not yet efficient. A method similar to the one
   * used in interp_gather should be used.
   */
  template <bool safe>
  std::array<VoxelData, 6> get_face_neighbors(const int x,
                                              const int y,
                                              const int z) const;

  /*! \brief Fetch the voxel block which contains voxel (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  VoxelBlock<T>* fetch(const int x, const int y, const int z) const;

  /*! \brief Fetch the node (x,y,z) at depth
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   * \param depth depth to be searched
   */
  Node<T>* fetch_node(const int x, const int y, const int z,
      const int depth) const;

  /*! \brief Insert the octant at (x,y,z). Not thread safe.
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   * \param depth target insertion depth
   */
  Node<T>* insert(const int x, const int y, const int z, const int depth);

  /*! \brief Insert the block (x,y,z) at maximum resolution. Not thread safe.
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  VoxelBlock<T>* insert(const int x, const int y, const int z);

  /*! \brief Interp voxel value at voxel position  (x,y,z)
   * \param voxel_coord_f three-dimensional coordinates in which each component belongs
   * to the interval [0, size]
   * \return signed distance function value at voxel position (x, y, z)
   */
  template <typename ValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               ValueSelector          select_value) const;

  /*! \brief Interp voxel value at voxel position  (x,y,z)
   * \param voxel_coord_f three-dimensional coordinates in which each component belongs
   * to the interval [0, size]
   * \param stride distance between neighbouring sampling point, in voxels
   * \return signed distance function value at voxel position (x, y, z)
   */

  template <typename ValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               const int              min_scale,
                               ValueSelector          select_value) const;

  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               NodeValueSelector      select_node_value,
                               VoxelValueSelector     select_voxel_value) const;

  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               const int              min_scale,
                               NodeValueSelector      select_node_value,
                               VoxelValueSelector     select_voxel_value) const;

  template <typename ValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               const int              min_scale,
                               ValueSelector          select_value,
                               bool&                  is_valid) const;

  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               const int              min_scale,
                               NodeValueSelector      select_node_value,
                               VoxelValueSelector     select_voxel_value,
                               bool&                  is_valid) const;


  /*! \brief Compute the gradient at voxel position  (x,y,z)
   * \param voxel_coord_f three-dimensional coordinates in which each component belongs
   * to the interval [0, size]
   * \return gradient at voxel position pos
   */
  template <typename FieldSelect>
  Eigen::Vector3f grad(const Eigen::Vector3f& voxel_coord_f, FieldSelect select_value) const;

  /*! \brief Compute gradient at voxel position  (x,y,z)
   * \param voxel_coord_f three-dimensional coordinates in which each component belongs
   * to the interval [0, _size]
   * \param stride distance between neighbouring sampling point, in voxels.
   * Must be >= 1
   * \return signed distance function value at voxel position (x, y, z)
   */
  template <typename FieldSelect>
  Eigen::Vector3f grad(const Eigen::Vector3f& voxel_coord_f, const int scale,
      FieldSelect select_value) const;

  /*! \brief Get the list of allocated block. If the active switch is set to
   * true then only the visible blocks are retrieved.
   * \param block_list output vector of allocated blocks
   * \param active boolean switch. Set to true to retrieve visible, allocated
   * blocks, false to retrieve all allocated blocks.
   */
  void getBlockList(std::vector<VoxelBlock<T> *>& block_list, bool active);
  typename T::template MemoryPoolType<T>& pool() { return pool_; };
  const typename T::template MemoryPoolType<T>& pool() const { return pool_; };

  /*! \brief Computes the morton code of the block containing voxel
   * at coordinates (x,y,z)
   * \param x x coordinate in interval [0, size]
   * \param y y coordinate in interval [0, size]
   * \param z z coordinate in interval [0, size]
   */
  key_t hash(const int x, const int y, const int z) {
    const int scale = voxel_depth_ - math::log2_const(block_size);
    return keyops::encode(x, y, z, scale, voxel_depth_);
  }

  key_t hash(const int x, const int y, const int z, key_t scale) {
    return keyops::encode(x, y, z, scale, voxel_depth_);
  }

  /*! \brief allocate a set of voxel blocks via their positional key
   * \param keys collection of voxel block keys to be allocated (i.e. their
   * morton number)
   * \param number of keys in the keys array
   */
  bool allocate(key_t *keys, int num_elem);

  void save(const std::string& filename);
  void load(const std::string& filename);

  /*! \brief Counts the number of blocks allocated
   * \return number of voxel blocks allocated
   */
  int blockCount();

  /*! \brief Counts the number of internal nodes
   * \return number of internal nodes
   */
  int nodeCount();

  void printMemStats(){
    // memory.printStats();
  };

private:

  Node<T>* root_;
  int size_;
  float dim_;
  int num_levels_;
  int voxel_depth_;
  int max_block_scale_;
  int block_depth_;
  typename T::template MemoryPoolType<T> pool_;

  friend class VoxelBlockRayIterator<T>;
  friend class node_iterator<T>;

  // Allocation specific variables
  key_t* keys_at_depth_;
  int reserved_;

  // Private implementation of cached methods
  VoxelData get(const int x, const int y, const int z, VoxelBlock<T>* cached) const;
  VoxelData get(const Eigen::Vector3f& voxel_coord_f, VoxelBlock<T>* cached) const;

  VoxelData get(const int x, const int y, const int z,
     int&  scale, VoxelBlock<T>* cached) const;
  VoxelData get(const Eigen::Vector3f& voxel_coord_f, int& scale,
      VoxelBlock<T>* cached) const;

  // Parallel allocation of a given tree depth for a set of input keys.
  // Pre: depth above target_depth must have been already allocated
  bool allocate_depth(key_t * keys, int num_tasks, int target_depth);

  void reserveBuffers(const int n);

  // General helpers

  int blockCountRecursive(Node<T>*);
  int nodeCountRecursive(Node<T>*);
  void getActiveBlockList(Node<T>*, std::vector<VoxelBlock<T> *>& block_list);
  void getAllocatedBlockList(Node<T>*, std::vector<VoxelBlock<T> *>& block_list);

  void deleteNode(Node<T>** node);
  void deallocateTree(){ deleteNode(&root_); }
};

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get(const Eigen::Vector3f& point_M,
    VoxelBlock<T>* cached) const {
  return get(point_M, 0, cached);
}

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get(const Eigen::Vector3f& point_M,
    int& scale, VoxelBlock<T>* cached) const {

  const Eigen::Vector3i voxel_coord = (point_M.homogeneous() *
      Eigen::Vector4f::Constant(size_ / dim_)).template head<3>().template cast<int>();

  if(cached != NULL){
    Eigen::Vector3i lower_coord = cached->coordinates();
    Eigen::Vector3i upper_coord = lower_coord + Eigen::Vector3i::Constant(block_size-1);
    const int contained =
      ((voxel_coord.array() >= lower_coord.array()) * (voxel_coord.array() <= upper_coord.array())).all();
    if(contained){
      return cached->data(voxel_coord, scale);
    }
  }

  Node<T>* node = root_;
  if(!node) {
    return T::invalid();
  }

  // Get the block.

  unsigned node_size = size_ >> 1;
  for(; node_size >= block_size; node_size = node_size >> 1){
    node = node->child((voxel_coord.x() & node_size) > 0,
                       (voxel_coord.y() & node_size) > 0,
                       (voxel_coord.z() & node_size) > 0);
    if(!node){
    return T::invalid();
    }
  }

  // Get the element in the voxel block
  auto block = static_cast<VoxelBlock<T>*>(node);
  scale = std::max(block->current_scale(), scale);
  return static_cast<VoxelBlock<T>*>(node)->data(voxel_coord, scale);
}

template <typename T>
inline void  Octree<T>::set(const int x,
    const int y, const int z, const VoxelData data) {

  Node<T>* node = root_;
  if(!node) {
    return;
  }

  unsigned node_size = size_ >> 1;
  for(; node_size >= block_size; node_size = node_size >> 1){
    Node<T>* node_tmp = node->child((x & node_size) > 0, (y & node_size) > 0, (z & node_size) > 0);
    if(!node_tmp){
      return;
    }
    node = node_tmp;
  }

  static_cast<VoxelBlock<T> *>(node)->data(Eigen::Vector3i(x, y, z), data);
}

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get(const int x,
    const int y, const int z) const {

  Node<T>* node = root_;
  if(!node) {
    return T::initData();
  }

  unsigned node_size = size_ >> 1;
  for(; node_size >= block_size; node_size = node_size >> 1){
    const int child_idx = ((x & node_size) > 0) +  2 * ((y & node_size) > 0) +  4*((z & node_size) > 0);
    Node<T>* node_tmp = node->child(child_idx);
    if(!node_tmp){
      return node->data_[child_idx];
    }
    node = node_tmp;
  }

  return static_cast<VoxelBlock<T> *>(node)->data(Eigen::Vector3i(x, y, z));
}

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get_fine(const int x,
    const int y, const int z, const int scale) const {
  assert(scale < voxel_depth_);

  Node<T>* node = root_;
  if(!node) {
    return T::initData();
  }


  const unsigned min_node_size = std::max((1 << scale), (int) block_size);
  unsigned node_size = size_ >> 1;
  int child_idx;
  for(; node_size >= min_node_size; node_size = node_size >> 1) {
    child_idx  = ((x & node_size) > 0) + 2 * ((y & node_size) > 0) + 4*((z & node_size) > 0);
    Node<T>* node_tmp = node->child(child_idx);
    if(!node_tmp){
      auto& value = node->data_[child_idx];
      return value;
    }
    node = node_tmp;
  }

  if(min_node_size == block_size) {
    auto block = static_cast<VoxelBlock<T> *>(node);
    return block->data(Eigen::Vector3i(x, y, z), std::max(scale, block->current_scale()));
  } else {
    return (node->parent())->data_[child_idx];
  }
}
template <typename T>
template <bool safe>
inline std::array<typename Octree<T>::VoxelData, 6> Octree<T>::get_face_neighbors(
    const int x,
    const int y,
    const int z) const {

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
        neighbor_data[i] = get_fine(neighbor_x, neighbor_y, neighbor_z);
      } else {
        // The neighbor voxel is outside the map, set the value to empty.
        neighbor_data[i] = T::invalid();
      }
    } else {
      // Get the value of the neighbor voxel.
      neighbor_data[i] = get_fine(neighbor_x, neighbor_y, neighbor_z);
    }
  }

  return neighbor_data;
}

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get(const int x,
   const int y, const int z, VoxelBlock<T>* cached) const {
  return get(x, y, z, 0, cached);
}

template <typename T>
inline typename Octree<T>::VoxelData Octree<T>::get(const int x,
   const int y, const int z, int& scale, VoxelBlock<T>* cached) const {

  if(cached != NULL){
    const Eigen::Vector3i voxel_coord = Eigen::Vector3i(x, y, z);
    const Eigen::Vector3i lower_coord = cached->coordinates();
    const Eigen::Vector3i upper_coord = lower_coord + Eigen::Vector3i::Constant(block_size - 1);
    const int contained =
      ((voxel_coord.array() >= lower_coord.array()) && (voxel_coord.array() <= upper_coord.array())).all();
    if(contained){
      scale = std::max(cached->current_scale(), scale);
      return cached->data(Eigen::Vector3i(x, y, z), scale);
    }
  }

  Node<T>* node  = root_;
  if(!node) {
    return T::initData();
  }

  unsigned node_size = size_ >> 1;
  for(; node_size >= block_size; node_size = node_size >> 1){
    node = node->child((x & node_size) > 0, (y & node_size) > 0, (z & node_size) > 0);
    if(!node){
      return T::initData();
    }
  }
  auto block = static_cast<VoxelBlock<T> *>(node);
  scale = std::max(block->current_scale(), scale);
  return block->data(Eigen::Vector3i(x, y, z), scale);
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
  voxel_depth_ = log2(size);
  num_levels_ = voxel_depth_ + 1;
  max_block_scale_ = log2(block_size);
  block_depth_ = voxel_depth_ - max_block_scale_;
  root_ = pool_.root();
  root_->size_ = size;
  reserved_ = 1024;
  keys_at_depth_ = new key_t[reserved_];
  std::memset(keys_at_depth_, 0, reserved_);
}

template <typename T>
inline VoxelBlock<T>* Octree<T>::fetch(const int x, const int y,
   const int z) const {

  Node<T>* node = root_;
  if(!node) {
    return NULL;
  }

  // Get the block.
  unsigned node_size = size_ / 2;
  for(; node_size >= block_size; node_size /= 2){
    node = node->child((x & node_size) > 0u, (y & node_size) > 0u, (z & node_size) > 0u);
    if(!node){
      return NULL;
    }
  }
  return static_cast<VoxelBlock<T>* > (node);
}

template <typename T>
inline Node<T>* Octree<T>::fetch_node(const int x, const int y,
   const int z, const int depth) const {

  Node<T>* node = root_;
  if(!node) {
    return NULL;
  }

  // Get the block.
  unsigned node_size = size_ / 2;
  for(int d = 1; node_size >= block_size && d <= depth; node_size /= 2, ++d){
    node = node->child((x & node_size) > 0u, (y & node_size) > 0u, (z & node_size) > 0u);
    if(!node){
      return NULL;
    }
  }
  return node;
}

template <typename T>
Node<T>* Octree<T>::insert(const int x, const int y, const int z,
    const int depth) {

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
    root_->code_ = 0;
    root_->size_ = size_;
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
        node_tmp = pool_.acquireBlock();
        node_tmp->parent() = node;
        static_cast<VoxelBlock<T> *>(node_tmp)->coordinates(
            Eigen::Vector3i(unpack_morton(prefix)));
        static_cast<VoxelBlock<T> *>(node_tmp)->active(true);
        static_cast<VoxelBlock<T> *>(node_tmp)->code_ = prefix | d;
        node->children_mask_ = node->children_mask_ | (1 << child_idx);
      } else {
        node_tmp = pool_.acquireNode();
        node_tmp->parent() = node;
        node_tmp->code_ = prefix | d;
        node_tmp->size_ = node_size;
        node->children_mask_ = node->children_mask_ | (1 << child_idx);
        // std::cout << "coords: "
        //   << keyops::decode(keyops::code(node_tmp->code_)) << std::endl;
      }
      node->child(child_idx) = node_tmp;
    }
    node = node_tmp;
  }
  return node;
}

template <typename T>
VoxelBlock<T>* Octree<T>::insert(const int x, const int y, const int z) {
  return static_cast<VoxelBlock<T> * >(insert(x, y, z, voxel_depth_));
}

template <typename T>
template <typename FieldSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord,
                                        FieldSelector          select_value) const {
  return interp(voxel_coord, 0, select_value, select_value);
}

template <typename T>
template <typename ValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord,
                                        const int              min_scale,
                                        ValueSelector          select_value) const {
  return interp(voxel_coord, min_scale, select_value, select_value);
}

template <typename T>
template <typename NodeValueSelector,
          typename VoxelValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        const int              min_scale,
                                        NodeValueSelector      select_node_value,
                                        VoxelValueSelector     select_voxel_value) const {

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
template <typename ValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        const int              min_scale,
                                        ValueSelector          select_value,
                                        bool&                  is_valid) const {
  return interp(voxel_coord_f, min_scale, select_value, select_value, is_valid);
}

template <typename T>
template <typename NodeValueSelector, typename VoxelValueSelector>
std::pair<float, int> Octree<T>::interp(const Eigen::Vector3f& voxel_coord_f,
                                        const int              min_scale,
                                        NodeValueSelector      select_node_value,
                                        VoxelValueSelector     select_voxel_value,
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

    int interp_scale = se::internal::gather_values(
        *this, base_coord, target_scale, select_node_value, select_voxel_value, voxel_values);
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
template <typename FieldSelector>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f, const int init_scale,
    FieldSelector select_value) const {

  int iter = 0;
  int scale = init_scale;
  int last_scale = scale;
  Eigen::Vector3f factor = Eigen::Vector3f::Constant(0);
  Eigen::Vector3f gradient = Eigen::Vector3f::Constant(0);
  while(iter < 3) {
    const int stride = 1 << scale;
    const Eigen::Vector3f scaled_voxel_coord_f = 1.f/stride * voxel_coord_f - sample_offset_frac_;
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


    VoxelBlock<T>* block = fetch(base_coord.x(), base_coord.y(), base_coord.z());
    gradient.x() = (((select_value(get(upper_lower_coord.x(), lower_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_lower_coord.x(), lower_coord.y(), lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_upper_coord.x(), lower_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_upper_coord.x(), lower_coord.y(), lower_coord.z(), scale, block))) * factor.x())
        * (1 - factor.y())
        + ((select_value(get(upper_lower_coord.x(), upper_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_lower_coord.x(), upper_coord.y(), lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_upper_coord.x(), upper_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_upper_coord.x(), upper_coord.y(), lower_coord.z(), scale, block)))
          * factor.x()) * factor.y()) * (1 - factor.z())
      + (((select_value(get(upper_lower_coord.x(), lower_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_lower_coord.x(), lower_coord.y(), upper_coord.z(), scale, block))) * (1 - factor.x())
            + (select_value(get(upper_upper_coord.x(), lower_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_upper_coord.x(), lower_coord.y(), upper_coord.z(), scale, block)))
            * factor.x()) * (1 - factor.y())
          + ((select_value(get(upper_lower_coord.x(), upper_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_lower_coord.x(), upper_coord.y(), upper_coord.z(), scale, block)))
            * (1 - factor.x())
            + (select_value(get(upper_upper_coord.x(), upper_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_upper_coord.x(), upper_coord.y(), upper_coord.z(), scale, block)))
            * factor.x()) * factor.y()) * factor.z();
    if(scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }

    gradient.y() = (((select_value(get(lower_coord.x(), upper_lower_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_coord.x(), lower_lower_coord.y(), lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_coord.x(), upper_lower_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(upper_coord.x(), lower_lower_coord.y(), lower_coord.z(), scale, block))) * factor.x())
        * (1 - factor.y())
        + ((select_value(get(lower_coord.x(), upper_upper_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(lower_coord.x(), lower_upper_coord.y(), lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_coord.x(), upper_upper_coord.y(), lower_coord.z(), scale, block))
            - select_value(get(upper_coord.x(), lower_upper_coord.y(), lower_coord.z(), scale, block)))
          * factor.x()) * factor.y()) * (1 - factor.z())
      + (((select_value(get(lower_coord.x(), upper_lower_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_coord.x(), lower_lower_coord.y(), upper_coord.z(), scale, block))) * (1 - factor.x())
            + (select_value(get(upper_coord.x(), upper_lower_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(upper_coord.x(), lower_lower_coord.y(), upper_coord.z(), scale, block)))
            * factor.x()) * (1 - factor.y())
          + ((select_value(get(lower_coord.x(), upper_upper_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(lower_coord.x(), lower_upper_coord.y(), upper_coord.z(), scale, block)))
            * (1 - factor.x())
            + (select_value(get(upper_coord.x(), upper_upper_coord.y(), upper_coord.z(), scale, block))
              - select_value(get(upper_coord.x(), lower_upper_coord.y(), upper_coord.z(), scale, block)))
            * factor.x()) * factor.y()) * factor.z();
    if(scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }

    gradient.z() = (((select_value(get(lower_coord.x(), lower_coord.y(), upper_lower_coord.z(), scale, block))
            - select_value(get(lower_coord.x(), lower_coord.y(), lower_lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_coord.x(), lower_coord.y(), upper_lower_coord.z(), scale, block))
            - select_value(get(upper_coord.x(), lower_coord.y(), lower_lower_coord.z(), scale, block))) * factor.x())
        * (1 - factor.y())
        + ((select_value(get(lower_coord.x(), upper_coord.y(), upper_lower_coord.z(), scale, block))
            - select_value(get(lower_coord.x(), upper_coord.y(), lower_lower_coord.z(), scale, block))) * (1 - factor.x())
          + (select_value(get(upper_coord.x(), upper_coord.y(), upper_lower_coord.z(), scale, block))
            - select_value(get(upper_coord.x(), upper_coord.y(), lower_lower_coord.z(), scale, block)))
          * factor.x()) * factor.y()) * (1 - factor.z())
      + (((select_value(get(lower_coord.x(), lower_coord.y(), upper_upper_coord.z(), scale, block))
              - select_value(get(lower_coord.x(), lower_coord.y(), lower_upper_coord.z(), scale, block))) * (1 - factor.x())
            + (select_value(get(upper_coord.x(), lower_coord.y(), upper_upper_coord.z(), scale, block))
              - select_value(get(upper_coord.x(), lower_coord.y(), lower_upper_coord.z(), scale, block)))
            * factor.x()) * (1 - factor.y())
          + ((select_value(get(lower_coord.x(), upper_coord.y(), upper_upper_coord.z(), scale, block))
              - select_value(get(lower_coord.x(), upper_coord.y(), lower_upper_coord.z(), scale, block)))
            * (1 - factor.x())
            + (select_value(get(upper_coord.x(), upper_coord.y(), upper_upper_coord.z(), scale, block))
              - select_value(get(upper_coord.x(), upper_coord.y(), lower_upper_coord.z(), scale, block)))
            * factor.x()) * factor.y()) * factor.z();
    if(scale != last_scale) {
      last_scale = scale;
      iter++;
      continue;
    }
    break;
  }

  return (0.5f * dim_ / size_) * gradient;
}

template <typename T>
template <typename FieldSelector>
Eigen::Vector3f Octree<T>::grad(const Eigen::Vector3f& voxel_coord_f, FieldSelector select_value) const {
  return grad(voxel_coord_f, 1, select_value);
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
    delete[] keys_at_depth_;
    keys_at_depth_ = new key_t[num_blocks];
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
    compute_prefix(keys, keys_at_depth_, num_elem, mask);
    last_elem = algorithms::unique_multiscale(keys_at_depth_, num_elem);
    success = allocate_depth(keys_at_depth_, last_elem, depth);
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
          (*node)->size_ = octant_size;
          static_cast<VoxelBlock<T> *>(*node)->coordinates(Eigen::Vector3i(unpack_morton(octant_key)));
          static_cast<VoxelBlock<T> *>(*node)->active(true);
          static_cast<VoxelBlock<T> *>(*node)->code_ = octant_key | depth;
          parent->children_mask_ = parent->children_mask_ | (1 << child_idx);
        } else {
          *node = pool_.acquireNode();
          (*node)->parent() = parent;
          (*node)->code_ = octant_key | depth;
          (*node)->size_ = octant_size;
          parent->children_mask_ = parent->children_mask_ | (1 << child_idx);
        }
      }
      octant_size /= 2;
    }
  }
  return true;
}

template <typename T>
void Octree<T>::getBlockList(std::vector<VoxelBlock<T>*>& block_list, bool active){
  Node<T>* node = root_;
  if(!node) return;
  if(active) getActiveBlockList(node, block_list);
  else getAllocatedBlockList(node, block_list);
}

template <typename T>
void Octree<T>::getActiveBlockList(Node<T>* node,
    std::vector<VoxelBlock<T>*>& block_list){
  if(!node) return;
  std::queue<Node<T> *> node_queue;
  node_queue.push(node);
  while(!node_queue.empty()){
    Node<T>* node_tmp = node_queue.front();
    node_queue.pop();

    if(node_tmp->isBlock()){
      VoxelBlock<T>* block = static_cast<VoxelBlock<T> *>(node_tmp);
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
    std::vector<VoxelBlock<T>*>& block_list){
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
  const int block_size_cube = se::VoxelBlock<T>::size_cube;

  is.read(reinterpret_cast<char *>(&size), sizeof(size));
  is.read(reinterpret_cast<char *>(&dim), sizeof(dim));

  init(size, dim);

  size_t num_nodes = 0;
  is.read(reinterpret_cast<char *>(&num_nodes), sizeof(size_t));
  pool_.reserveNodes(num_nodes);
  std::cout << "Reading " << num_nodes << " nodes " << std::endl;
  for(size_t i = 0; i < num_nodes; ++i) {
    Node<T> node;
    internal::deserialise(node, is);
    Eigen::Vector3i node_coord = keyops::decode(node.code_);
    Node<T>* node_ptr = insert(node_coord.x(), node_coord.y(), node_coord.z(), keyops::depth(node.code_));
    node_ptr->timestamp(node.timestamp());
    std::memcpy(node_ptr->data_, node.data_, 8 * sizeof(VoxelData));
  }

  size_t num_blocks = 0;
  is.read(reinterpret_cast<char *>(&num_blocks), sizeof(size_t));
  std::cout << "Reading " << num_blocks << " blocks " << std::endl;
  for(size_t i = 0; i < num_blocks; ++i) {
    VoxelBlock<T> block;
    internal::deserialise(block, is);
    Eigen::Vector3i block_coord = block.coordinates();
    VoxelBlock<T>* block_ptr =
      static_cast<VoxelBlock<T> *>(insert(block_coord.x(), block_coord.y(), block_coord.z(), keyops::depth(block.code_)));
    block_ptr->min_scale(block.min_scale());
    block_ptr->current_scale(block.current_scale());
    std::memcpy(block_ptr->getBlockRawPtr(), block.getBlockRawPtr(), (block_size_cube + 64 + 8 + 1) * sizeof(*(block.getBlockRawPtr())));
  }
}
}
template <typename FieldType>
const Eigen::Vector3f se::Octree<FieldType>::sample_offset_frac_ =
  Eigen::Vector3f::Constant(SAMPLE_POINT_POSITION);
#endif // OCTREE_H
