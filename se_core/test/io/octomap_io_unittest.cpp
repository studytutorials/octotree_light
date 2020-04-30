// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#if defined(SE_OCTOMAP) && SE_OCTOMAP

#include "gtest/gtest.h"

#include <memory>

#include "se/octree.hpp"
#include "se/io/octomap_io.hpp"



struct Voxel {
  struct VoxelData {
    float x;
    double y;
  };

  static inline VoxelData invalid(){ return {0.f, 0.0}; }
  static inline VoxelData initData(){ return {0.f, 0.0}; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



class OctoMapIO : public ::testing::Test {
  protected:
    virtual void SetUp() {
      // Initialize the octrees.
      octree_uninitialized_ = std::unique_ptr<se::Octree<Voxel>>(new se::Octree<Voxel>);
      octree_unallocated_ = std::unique_ptr<se::Octree<Voxel>>(new se::Octree<Voxel>);
      octree_unallocated_->init(octree_size_, octree_dim_);
      octree_unknown_ = std::unique_ptr<se::Octree<Voxel>>(new se::Octree<Voxel>);
      octree_unknown_->init(octree_size_, octree_dim_);
      octree_ = std::unique_ptr<se::Octree<Voxel>>(new se::Octree<Voxel>);
      octree_->init(octree_size_, octree_dim_);

      // Allocate some VoxelBlocks/Nodes.
      constexpr size_t num_blocks = 16;
      constexpr size_t num_nodes = 4;
      const Eigen::Vector3i nodes[num_blocks + num_nodes] = {
          { 0,  0,  0}, { 8,  0,  0}, {16,  0,  0}, {24,  0,  0},
          { 0,  8,  0}, { 8,  8,  0}, {16,  8,  0}, {24,  8,  0},
          { 0, 16,  0}, { 8, 16,  0}, {16, 16,  0}, {24, 16,  0},
          { 0, 24,  0}, { 8, 24,  0}, {16, 24,  0}, {24, 24,  0},
          { 0,  0, 16}, {16,  0, 16}, { 0, 16, 16}, {16, 16, 16}};
      const int levels[num_blocks + num_nodes] = {
          2, 2, 2, 2,
          2, 2, 2, 2,
          2, 2, 2, 2,
          2, 2, 2, 2,
          1, 1, 1, 1};
      se::key_t alloc_list[num_blocks + num_nodes];
      for (size_t i = 0; i < num_blocks + num_nodes; ++i) {
        alloc_list[i] = octree_->hash(nodes[i](0), nodes[i](1), nodes[i](2), levels[i]);
      }
      octree_unknown_->allocate(alloc_list, num_blocks + num_nodes);
      octree_->allocate(alloc_list, num_blocks + num_nodes);

      // Set the data of all voxels.
      for (int z = 0; z < 8; ++z) {
        for (int y = 0; y < octree_size_; ++y) {
          for (int x = 0; x < octree_size_; ++x) {
            const Voxel::VoxelData voxel_data = {(z - 4) * value_increment_, 0.0};
            octree_->set(x, y, z, voxel_data);
            num_updated_voxels++;
          }
        }
      }
    }

    std::unique_ptr<se::Octree<Voxel>> octree_uninitialized_;
    std::unique_ptr<se::Octree<Voxel>> octree_unallocated_;
    std::unique_ptr<se::Octree<Voxel>> octree_unknown_;
    std::unique_ptr<se::Octree<Voxel>> octree_;
    const int octree_size_ = 32;
    const float octree_dim_ = 1.f;
    const float voxel_dim_ = octree_dim_ / octree_size_;
    const float value_increment_ = 0.05;
    size_t num_updated_voxels = 0;
};



// Convert an uninitialized se::Octree to an OctoMap.
TEST_F(OctoMapIO, ToOctoMapUninitialized) {
  std::unique_ptr<octomap::OcTree> omap (se::to_octomap(*octree_uninitialized_));
  ASSERT_EQ(omap, nullptr);
  std::unique_ptr<octomap::OcTree> omap_binary (se::to_binary_octomap(*octree_uninitialized_));
  ASSERT_EQ(omap_binary, nullptr);
}



// Convert an unallocated se::Octree to an OctoMap.
TEST_F(OctoMapIO, ToOctoMapUnallocated) {
  std::unique_ptr<octomap::OcTree> omap (se::to_octomap(*octree_unallocated_));
  ASSERT_NE(omap, nullptr);
  ASSERT_EQ(omap->size(), 0);
  std::unique_ptr<octomap::OcTree> omap_binary (se::to_binary_octomap(*octree_unallocated_));
  ASSERT_NE(omap_binary, nullptr);
  ASSERT_EQ(omap_binary->size(), 0);
}



// Convert an unknown (all allocated voxels have an occupancy probability of
// 0.5) se::Octree to an OctoMap.
TEST_F(OctoMapIO, ToOctoMapUnknown) {
  std::unique_ptr<octomap::OcTree> omap (se::to_octomap(*octree_unknown_));
  ASSERT_NE(omap, nullptr);
  ASSERT_EQ(omap->size(), 0);
  std::unique_ptr<octomap::OcTree> omap_binary (se::to_binary_octomap(*octree_unknown_));
  ASSERT_NE(omap_binary, nullptr);
  ASSERT_EQ(omap_binary->size(), 0);
}



// Convert an se::Octree to an OctoMap.
TEST_F(OctoMapIO, ToOctoMap) {
  std::unique_ptr<octomap::OcTree> omap (se::to_octomap(*octree_));
  ASSERT_NE(omap, nullptr);
  ASSERT_FLOAT_EQ(omap->getResolution(), voxel_dim_);
  double dim_x = 0.0;
  double dim_y = 0.0;
  double dim_z = 0.0;
  omap->getMetricSize(dim_x, dim_y, dim_z);
  ASSERT_FLOAT_EQ(dim_x, octree_dim_);
  ASSERT_FLOAT_EQ(dim_y, octree_dim_);
  ASSERT_FLOAT_EQ(dim_z, octree_dim_ / 4.f);
  // TODO More thorough tests:
  // - How to compute the expected number of voxels? OctoMap prunes voxels with
  //   the same data, but supereight does not.
  // - Test the individual voxel data in OctoMap and supereight. Can't seem to
  //   find an OctoMap function to get the data of a single voxel.
  ASSERT_TRUE(omap->writeBinary("/tmp/octomap.bt"));

  std::unique_ptr<octomap::OcTree> omap_binary (se::to_binary_octomap(*octree_));
  ASSERT_NE(omap_binary, nullptr);
  ASSERT_FLOAT_EQ(omap_binary->getResolution(), voxel_dim_);
  dim_x = 0.0;
  dim_y = 0.0;
  dim_z = 0.0;
  omap->getMetricSize(dim_x, dim_y, dim_z);
  ASSERT_FLOAT_EQ(dim_x, octree_dim_);
  ASSERT_FLOAT_EQ(dim_y, octree_dim_);
  ASSERT_FLOAT_EQ(dim_z, octree_dim_ / 4.f);
  // TODO More thorough tests:
  // - How to compute the expected number of voxels? OctoMap prunes voxels with
  //   the same data, but supereight does not.
  // - Test the individual voxel data in OctoMap and supereight. Can't seem to
  //   find an OctoMap function to get the data of a single voxel.
  ASSERT_TRUE(omap_binary->writeBinary("/tmp/octomap_binary.bt"));
}

#endif

