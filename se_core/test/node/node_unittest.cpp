// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include <se/node.hpp>
#include <se/octree.hpp>



struct TestVoxelT {
  typedef struct VoxelData {
    float x;
    float y;
  } VoxelData;

  static inline VoxelData invalid(){ return {0.0f, 0.0f}; }
  static inline VoxelData initData(){ return {1.0f, 0.0f}; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;
  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



class NodeTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_ = std::make_unique<OctreeF>();
      octree_->init(size_, dim_);

      // Allocate all VoxelBlocks
      constexpr size_t num_voxel_blocks = se::math::cu(size_ / OctreeF::block_size);
      se::key_t allocation_list[num_voxel_blocks];
      int i = 0;
      for (int z = 0; z < size_; z += OctreeF::block_size) {
        for (int y = 0; y < size_; y += OctreeF::block_size) {
          for (int x = 0; x < size_; x += OctreeF::block_size) {
            allocation_list[i++] = octree_->hash(x, y, z);
          }
        }
      }
      octree_->allocate(allocation_list, num_voxel_blocks);
    }

    typedef se::Octree<TestVoxelT> OctreeF;
    std::unique_ptr<OctreeF> octree_;
    static constexpr int size_ = 32;
    static constexpr float dim_ = 32.0f;
    static constexpr float voxel_dim_ = dim_ / size_;
    static constexpr float block_dim_ = voxel_dim_ * OctreeF::block_size;
};



TEST_F(NodeTest, coordinates) {
  Eigen::Matrix<int, 3, 8> expected_node_coord;
  expected_node_coord << 0, 16,  0, 16,  0, 16,  0, 16,
                         0,  0, 16, 16,  0,  0, 16, 16,
                         0,  0,  0,  0, 16, 16, 16, 16;
  const se::Node<TestVoxelT>* root = octree_->root();
  for (int child_idx = 0; child_idx < 8; ++child_idx) {
    const Eigen::Vector3i node_coord = root->child(child_idx)->coordinates();
    EXPECT_EQ(node_coord, expected_node_coord.col(child_idx));
  }
}

TEST_F(NodeTest, centreCoordinates) {
  Eigen::Matrix<int, 3, 8> expected_node_coord;
  expected_node_coord << 8, 24,  8, 24,  8, 24,  8, 24,
                         8,  8, 24, 24,  8,  8, 24, 24,
                         8,  8,  8,  8, 24, 24, 24, 24;
  const se::Node<TestVoxelT>* root = octree_->root();
  for (int child_idx = 0; child_idx < 8; ++child_idx) {
    const Eigen::Vector3i node_coord = root->child(child_idx)->centreCoordinates();
    EXPECT_EQ(node_coord, expected_node_coord.col(child_idx));
  }
}

TEST_F(NodeTest, childCoord) {
  Eigen::Matrix<int, 3, 8> expected_child_coord;
  expected_child_coord << 0, 16,  0, 16,  0, 16,  0, 16,
                          0,  0, 16, 16,  0,  0, 16, 16,
                          0,  0,  0,  0, 16, 16, 16, 16;
  const se::Node<TestVoxelT>* root = octree_->root();
  for (int child_idx = 0; child_idx < 8; ++child_idx) {
    const Eigen::Vector3i child_coord = root->childCoord(child_idx);
    EXPECT_EQ(child_coord, expected_child_coord.col(child_idx));
  }
}

TEST_F(NodeTest, childCentreCoord) {
  Eigen::Matrix<int, 3, 8> expected_child_coord;
  expected_child_coord << 8, 24,  8, 24,  8, 24,  8, 24,
                          8,  8, 24, 24,  8,  8, 24, 24,
                          8,  8,  8,  8, 24, 24, 24, 24;
  const se::Node<TestVoxelT>* root = octree_->root();
  for (int child_idx = 0; child_idx < 8; ++child_idx) {
    const Eigen::Vector3i child_coord = root->childCentreCoord(child_idx);
    EXPECT_EQ(child_coord, expected_child_coord.col(child_idx));
  }
}

