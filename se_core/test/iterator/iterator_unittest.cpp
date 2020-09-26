// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <algorithm>
#include <list>
#include <memory>

#include <se/octree.hpp>
#include <se/octree_iterator.hpp>



struct TestVoxel {
  typedef struct VoxelData {
    float x;
    float y;

    bool operator==(const VoxelData& other) const {
      return (x == other.x) && (y == other.y);
    }

    bool operator!=(const VoxelData& other) const {
      return !(*this == other);
    }
  } VoxelData;

  static inline VoxelData invalid(){ return {0.0f, 0.0f}; }
  static inline VoxelData initData(){ return {1.0f, 0.0f}; }

  static bool isValid(const VoxelData& data) {
    return (data.y > 0);
  };

  using VoxelBlockType = se::VoxelBlockFull<TestVoxel>;
  using MemoryPoolType = se::PagedMemoryPool<TestVoxel>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



// Initialize an octree containing:
// - Leaf Nodes without data
// - Leaf Nodes with data
// - Unallocated VoxelBlocks without data
// - Unallocated VoxelBlocks with data
// - VoxelBlocks with last integration scales 0, 1, 2 and 3
class OctreeIteratorTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_ = std::make_unique<OctreeF>();
      // VoxelBlocks: 4x4x4 = 64
      // Max depth: 2
      octree_->init(size_, dim_);

      // Set the data for 2 unallocated leaf Nodes at depth 1
      // Expecting 2  Volumes of size 16
      for (int child_idx = 5; child_idx < 8; child_idx += 2) {
        se::Node<TestVoxel>* parent = octree_->root();
        constexpr float dim = dim_ / 2.0f;
        constexpr int size = size_ / 2;
        constexpr TestVoxel::VoxelData data {5.0f, 1.0f};
        const Eigen::Vector3i node_centre_coord = parent->childCentreCoord(child_idx);
        const Eigen::Vector3f centre_M = voxel_dim_ * node_centre_coord.cast<float>();
        // Set the data
        parent->childData(child_idx, data);
        // Add the volume to the list of expected volumes
        volumes_.emplace(volumes_.end(), centre_M, dim, size, data);
      }

      // Allocate 2 leaf Nodes at depth 1
      // Expecting 2 * 8 = 16 Volumes of size 8
      for (int y = 0; y < size_; y += 2 * block_size_) {
        // Allocate the Node
        constexpr int x = 0;
        constexpr int z = size_ / 2;
        constexpr TestVoxel::VoxelData data {4.0f, 1.0f};
        constexpr int size = block_size_;
        se::Node<TestVoxel>* node = octree_->insert(x, y, z, 1);
        // Set the data for all unallocated VoxelBlocks
        for (int i = 0; i < 8; ++i) {
          node->childData(i, data);
          // Add the expected volume
          const Eigen::Vector3i vb_centre_coord = node->childCentreCoord(i);
          const Eigen::Vector3f centre_M = voxel_dim_ * vb_centre_coord.cast<float>();
          volumes_.emplace(volumes_.end(), centre_M, block_dim_, size, data);
        }
      }

      // Allocate the bottom 16 VoxelBlocks
      // x, y in [0, size_ - 8], z = 0
      constexpr size_t num_voxel_blocks = 16;
      se::key_t allocation_list[num_voxel_blocks];
      int i = 0;
      for (int y = 0; y < size_; y += block_size_) {
        for (int x = 0; x < size_; x += block_size_) {
          allocation_list[i++] = octree_->hash(x, y, 0);
        }
      }
      octree_->allocate(allocation_list, num_voxel_blocks);
      // Set the individual voxel values
      for (int y = 0; y < size_; y += block_size_) {
        for (int x = 0; x < size_; x += block_size_) {
          // Compute the scale at which data will be set
          const int scale = y / block_size_;
          const TestVoxel::VoxelData data {static_cast<float>(scale), 1.0f};
          const int size = se::VoxelBlock<TestVoxel>::scaleVoxelSize(scale);
          const float dim = voxel_dim_ * size;
          // Fetch the VoxelBlock
          TestVoxel::VoxelBlockType* block = octree_->fetch(x, y, 0);
          // Set the last scale
          block->current_scale(scale);
          // Loop over all voxels at this scale
          for (int voxel_idx = 0; voxel_idx < block->scaleNumVoxels(scale); ++voxel_idx) {
            block->setData(voxel_idx, scale, data);
            const Eigen::Vector3i volume_coord = block->voxelCoordinates(voxel_idx, scale);
            const Eigen::Vector3f centre_coord_f = volume_coord.cast<float>() + Eigen::Vector3f::Constant(size / 2.0f);
            const Eigen::Vector3f centre_M = voxel_dim_ * centre_coord_f;
            volumes_.emplace(volumes_.end(), centre_M, dim, size, data);
          }
        }
      }
    }

    typedef se::Octree<TestVoxel> OctreeF;
    typedef se::OctreeIterator<TestVoxel> OctreeIterator;
    std::unique_ptr<OctreeF> octree_;
    static constexpr int size_ = 32;
    static constexpr float dim_ = 32.0f;
    static constexpr float voxel_dim_ = dim_ / size_;
    static constexpr int block_size_ = OctreeF::block_size;
    static constexpr float block_dim_ = voxel_dim_ * block_size_;
    std::list<se::Volume<TestVoxel> > volumes_;
};

constexpr int OctreeIteratorTest::size_;
constexpr float OctreeIteratorTest::dim_;
constexpr float OctreeIteratorTest::voxel_dim_;
constexpr int OctreeIteratorTest::block_size_;
constexpr float OctreeIteratorTest::block_dim_;



TEST_F(OctreeIteratorTest, Iterate) {
  for (auto v : *octree_) {
    // Search for the returned volume in volumes_. We don't care about the
    // order the volumes are returned in, hence the use of std::find()
    auto it = std::find(volumes_.begin(), volumes_.end(), v);
    // Erase if it has been found
    if (it != volumes_.end()) {
      volumes_.erase(it);
    }
    EXPECT_NE(it, volumes_.end());
  }
  ASSERT_EQ(volumes_.size(), 0u);
}

