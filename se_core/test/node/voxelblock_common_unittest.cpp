// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <vector>

#include<Eigen/StdVector>

#include <se/node.hpp>



struct TestVoxelT {
  typedef struct VoxelData {
    int x;
    int y;

    bool operator==(const VoxelData& other) const {
      return (x == other.x) && (y == other.y);
    }

    bool operator!=(const VoxelData& other) const {
      return !(*this == other);
    }
  } VoxelData;

  static inline VoxelData invalid(){ return {0, 0}; }
  static inline VoxelData initData(){ return {1, 0}; }
};



class VoxelBlockCommonTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      // Initialize the VoxelBlocks
      block_full_.size(size_);
      block_single_.size(size_);
      block_full_.coordinates(coordinates_);
      block_single_.coordinates(coordinates_);
      // Compute the number of voxels per scale
      for (int scale = 0; scale < num_scales_; ++scale) {
        voxels_per_scale_.push_back(se::VoxelBlockFull<TestVoxelT>::scaleNumVoxels(scale));
      }
      total_voxels_ = std::accumulate(voxels_per_scale_.begin(), voxels_per_scale_.end(), 0);
      // Compute all the voxel coordinates for all scales
      for (int scale = 0; scale < num_scales_; ++scale) {
        const int scale_voxel_size = se::VoxelBlockFull<TestVoxelT>::scaleVoxelSize(scale);
        for (int z = 0; z < size_; z += scale_voxel_size) {
          for (int y = 0; y < size_; y += scale_voxel_size) {
            for (int x = 0; x < size_; x += scale_voxel_size) {
              all_voxel_coords_.push_back(coordinates_ + Eigen::Vector3i(x, y, z));
            }
          }
        }
      }
    }

    typedef std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> Vector3iVector;

    static constexpr int size_ = se::VoxelBlock<TestVoxelT>::size_li;
    const Eigen::Vector3i coordinates_ = Eigen::Vector3i(3, 2, 1);
    const Eigen::Vector3i max_coordinates_ = coordinates_ + Eigen::Vector3i::Constant(size_);
    se::VoxelBlockFull<TestVoxelT> block_full_;
    se::VoxelBlockSingle<TestVoxelT> block_single_;
    static constexpr int num_scales_ = se::VoxelBlock<TestVoxelT>::max_scale + 1;
    std::vector<int> voxels_per_scale_;
    Vector3iVector all_voxel_coords_;
    int total_voxels_;
};

constexpr int VoxelBlockCommonTest::size_;
constexpr int VoxelBlockCommonTest::num_scales_;


TEST_F(VoxelBlockCommonTest, setAndGetData) {
  // Set data at all scales
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      block_full_.setData(voxel_idx, scale, data);
      block_single_.setDataSafe(voxel_idx, scale, data);
    }
  }
  // Read the data using the VoxelBlock method
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      EXPECT_EQ(block_full_.data(voxel_idx, scale), data);
      EXPECT_EQ(block_single_.data(voxel_idx, scale), data);
    }
  }
  // Read the data using a linear index
  int scale_offset = 0;
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const TestVoxelT::VoxelData data {scale, 1};
      EXPECT_EQ(block_full_.data(scale_offset + voxel_idx), data);
      EXPECT_EQ(block_single_.data(scale_offset + voxel_idx), data);
    }
    scale_offset += voxels_per_scale_[scale];
  }
}

TEST_F(VoxelBlockCommonTest, voxelCoordinates) {
  int scale_offset = 0;
  for (int scale = 0; scale < num_scales_; ++scale) {
    for (int voxel_idx = 0; voxel_idx < voxels_per_scale_[scale]; ++voxel_idx) {
      const int voxel_idx_li = scale_offset + voxel_idx;
      // Use relative linear index and scale
      const Eigen::Vector3i c_full = block_full_.voxelCoordinates(voxel_idx, scale);
      const Eigen::Vector3i c_single = block_single_.voxelCoordinates(voxel_idx, scale);
      EXPECT_EQ(c_full, all_voxel_coords_[voxel_idx_li]);
      EXPECT_EQ(c_single, all_voxel_coords_[voxel_idx_li]);
      // Use absolute linear index
      const Eigen::Vector3i c_full_li = block_full_.voxelCoordinates(voxel_idx_li);
      const Eigen::Vector3i c_single_li = block_single_.voxelCoordinates(voxel_idx_li);
      EXPECT_EQ(c_full_li, all_voxel_coords_[voxel_idx_li]);
      EXPECT_EQ(c_single_li, all_voxel_coords_[voxel_idx_li]);
    }
    scale_offset += voxels_per_scale_[scale];
  }
}

