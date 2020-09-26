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

#ifndef NODE_H
#define NODE_H

#include <time.h>
#include <atomic>
#include <vector>
#include "octree_defines.h"
#include "utils/math_utils.h"
#include "io/se_serialise.hpp"

namespace se {

static inline Eigen::Vector3f get_sample_coord(const Eigen::Vector3i& octant_coord,
                                             const int              octant_size,
                                             const Eigen::Vector3f& sample_offset_frac) {
  return octant_coord.cast<float>() + sample_offset_frac * octant_size;
}

/*! \brief A non-leaf node of the Octree. Each Node has 8 children.
 */
template <typename T>
class Node {

public:
  typedef typename T::VoxelData VoxelData;

  Node(const typename T::VoxelData init_data = T::initData());

  Node(const Node<T>& node);

  void operator=(const Node<T>& node);

  virtual ~Node(){};

  const VoxelData& data() const;

  VoxelData* childrenData() { return children_data_;}
  const VoxelData* childrenData() const { return children_data_;}
  VoxelData& childData(const int child_idx) { return children_data_[child_idx];}
  void childData(const int child_idx, const VoxelData& child_data) {
    children_data_[child_idx] = child_data;
  };

  Node*& child(const int x, const int y, const int z) {
    return children_ptr_[x + y * 2 + z * 4];
  };
  const Node* child(const int x, const int y, const int z) const {
    return children_ptr_[x + y * 2 + z * 4];
  };
  Node*& child(const int child_idx ) { return children_ptr_[child_idx]; }
  const Node* child(const int child_idx ) const { return children_ptr_[child_idx]; }

  Node*& parent() { return parent_ptr_; }
  const Node* parent() const { return parent_ptr_; }

  void code(key_t code) { code_ = code; }
  key_t code() const { return code_; }

  void size(int size) { size_ = size; }
  int size() const { return size_; }

  Eigen::Vector3i coordinates() const;
  Eigen::Vector3i centreCoordinates() const;

  Eigen::Vector3i childCoord(const int child_idx) const;
  Eigen::Vector3i childCentreCoord(const int child_idx) const;

  void children_mask(const unsigned char cm) { children_mask_ = cm; }
  unsigned char children_mask() const { return children_mask_; }

  void timestamp(const unsigned int t) { timestamp_ = t; }
  unsigned int timestamp() const { return timestamp_; }

  void active(const bool a){ active_ = a; }
  bool active() const { return active_; }

  virtual bool isBlock() const { return false; }

protected:
  VoxelData children_data_[8];
  Node* children_ptr_[8];
  Node* parent_ptr_;
  key_t code_;
  unsigned int size_;
  unsigned char children_mask_;
  unsigned int timestamp_;
  bool active_;

private:
  // Internal copy helper function
  void initFromNode(const Node<T>& node);
  friend std::ofstream& internal::serialise <> (std::ofstream& out, Node& node);
  friend void internal::deserialise <> (Node& node, std::ifstream& in);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlock: public Node<T> {

public:
  using VoxelData = typename T::VoxelData;

  static constexpr unsigned int size_li   = BLOCK_SIZE;
  static constexpr unsigned int size_sq   = se::math::sq(size_li);
  static constexpr unsigned int size_cu   = se::math::cu(size_li);
  static constexpr unsigned int max_scale = se::math::log2_const(size_li);

  VoxelBlock(const int current_scale,
             const int min_scale);

  VoxelBlock(const VoxelBlock<T>& block);

  void operator=(const VoxelBlock<T>& block);

  virtual ~VoxelBlock() {};

  bool isBlock() const { return true; }

  Eigen::Vector3i coordinates() const { return coordinates_; }
  void coordinates(const Eigen::Vector3i& block_coord){ coordinates_ = block_coord; }

  Eigen::Vector3i voxelCoordinates(const int voxel_idx) const;
  Eigen::Vector3i voxelCoordinates(const int voxel_idx, const int scale) const;

  int current_scale() const { return current_scale_; }
  void current_scale(const int s) { current_scale_ = s; }

  int min_scale() const { return min_scale_; }
  void min_scale(const int s) { min_scale_ = s; }

  virtual VoxelData data(const Eigen::Vector3i& voxel_coord) const = 0;
  virtual void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data) = 0;

  virtual VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const = 0;
  virtual void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data) = 0;

  virtual VoxelData data(const int voxel_idx) const = 0;
  virtual void setData(const int voxel_idx, const VoxelData& voxel_data) = 0;

  virtual VoxelData data(const int voxel_idx, const int scale) const = 0;
  virtual void setData(const int voxel_idx, const int scale, const VoxelData& voxel_data) = 0;

  /*! \brief The number of voxels per side at scale.
   */
  static constexpr int scaleSize(const int scale);
  /*! \brief The side length of a voxel at scale expressed in primitive voxels.
   * This is e.g. 1 for scale 0 and 8 for scale 3.
   */
  static constexpr int scaleVoxelSize(const int scale);
  /*! \brief The total number of voxels contained in scale.
   * This is equivalent to scaleSize()^3.
   */
  static constexpr int scaleNumVoxels(const int scale);
  /*! \brief The offset needed to get to the first voxel of scale when using
   * a linear index.
   */
  static constexpr int scaleOffset(const int scale);

protected:
  Eigen::Vector3i coordinates_;
  int current_scale_;
  int min_scale_;

private:
  // Internal copy helper function
  void initFromBlock(const VoxelBlock<T>& block);
};

/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockFinest: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockFinest(const typename T::VoxelData init_data = T::initData());

  VoxelBlockFinest(const VoxelBlockFinest<T>& block);

  void operator=(const VoxelBlockFinest<T>& block);

  virtual ~VoxelBlockFinest() {};

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  /**
   * \note Data will always retrieved and set at scale 0.
   *       Function only exist to keep API consistent.
   */
  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);

  /**
   * \note WARNING this functions should not be used with VoxelBlockFinest and will always just access at scale 0.
   *
   */
  VoxelData data(const int voxel_idx, const int scale) const;
  void setData(const int voxel_idx, const int scale, const VoxelData& voxel_data);

  VoxelData* blockData() { return block_data_; }
  const VoxelData* blockData() const { return block_data_; }
  static constexpr int data_size() { return sizeof(VoxelBlockFinest<T>); }

private:
  // Internal copy helper function
  void initFromBlock(const VoxelBlockFinest<T>& block);

  static constexpr size_t num_voxels_in_block = VoxelBlock<T>::size_cu;
  VoxelData block_data_[num_voxels_in_block]; // Brick of data.

  friend std::ofstream& internal::serialise <> (std::ofstream& out,
                                                VoxelBlockFinest& node);
  friend void internal::deserialise <> (VoxelBlockFinest& node, std::ifstream& in);
};



/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockFull: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockFull(const typename T::VoxelData init_data = T::initData());

  VoxelBlockFull(const VoxelBlockFull<T>& block);

  void operator=(const VoxelBlockFull<T>& block);

  virtual ~VoxelBlockFull() {};

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx, const int scale) const;
  void setData(const int voxel_idx, const int scale, const VoxelData& voxel_data);

  VoxelData* blockData() { return block_data_; }
  const VoxelData* blockData() const { return block_data_; }
  static constexpr int data_size() { return sizeof(VoxelBlockFull<T>); }

private:
  // Internal copy helper function
  void initFromBlock(const VoxelBlockFull<T>& block);

  static constexpr size_t compute_num_voxels() {
    size_t voxel_count = 0;
    unsigned int size_at_scale = VoxelBlock<T>::size_li;
    while(size_at_scale >= 1) {
      voxel_count += size_at_scale * size_at_scale * size_at_scale;
      size_at_scale = size_at_scale >> 1;
    }
    return voxel_count;
  }

  static constexpr size_t num_voxels_in_block = compute_num_voxels();
  VoxelData block_data_[num_voxels_in_block]; // Brick of data.

  friend std::ofstream& internal::serialise <> (std::ofstream& out,
                                                VoxelBlockFull& node);
  friend void internal::deserialise <> (VoxelBlockFull& node, std::ifstream& in);
};



/*! \brief A leaf node of the Octree. Each VoxelBlock contains compute_num_voxels_in_block() voxels
 * voxels.
 */
template <typename T>
class VoxelBlockSingle: public VoxelBlock<T> {

public:
  using VoxelData = typename VoxelBlock<T>::VoxelData;

  VoxelBlockSingle(const typename T::VoxelData init_data = T::initData());

  VoxelBlockSingle(const VoxelBlockSingle<T>& block);

  void operator=(const VoxelBlockSingle<T>& block);

  ~VoxelBlockSingle();

  VoxelData initData() const;
  void setInitData(const VoxelData& init_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord) const;
  void setData(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);
  void setDataSafe(const Eigen::Vector3i& voxel_coord, const VoxelData& voxel_data);

  VoxelData data(const Eigen::Vector3i& voxel_coord, const int scale) const;
  void setData(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);
  void setDataSafe(const Eigen::Vector3i& voxel_coord, const int scale, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx) const;
  void setData(const int voxel_idx, const VoxelData& voxel_data);
  void setDataSafe(const int voxel_idx, const VoxelData& voxel_data);

  VoxelData data(const int voxel_idx, const int scale) const;
  void setData(const int voxel_idx, const int scale, const VoxelData& voxel_data);
  void setDataSafe(const int voxel_idx, const int scale, const VoxelData& voxel_data);

  void allocateDownTo();
  void allocateDownTo(const int scale);

  void deleteUpTo(const int scale);

  std::vector<VoxelData*>& blockData() { return block_data_; }
  const std::vector<VoxelData*>& blockData() const { return block_data_; }
  static constexpr int data_size() { return sizeof(VoxelBlock<T>); }

private:
  // Internal copy helper function
  void initFromBlock(const VoxelBlockSingle<T>& block);
  void initialiseData(VoxelData* voxel_data, const int num_voxels);
  std::vector<VoxelData*> block_data_; // block_data_[0] returns the data at scale = max_scale and not scale = 0
  VoxelData init_data_;

  friend std::ofstream& internal::serialise <> (std::ofstream& out, VoxelBlockSingle& node);
  friend void internal::deserialise <> (VoxelBlockSingle& node, std::ifstream& in);
};

} // namespace se

#include "node_impl.hpp"

#endif
