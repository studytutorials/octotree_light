/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <array>

#include <gtest/gtest.h>

#include <se/neighbors/neighbor_gather.hpp>
#include <se/octree.hpp>
#include <se/utils/math_utils.h>

// Create a voxel trait storing a single float value.
struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



// Initialize an octree and store some values inside.
class NeighborGatherTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(64, 1);

      // Allocate some VoxelBlocks.
      constexpr size_t num_voxel_blocks = 1;
      const Eigen::Vector3i blocks_coord[num_voxel_blocks] = {{0, 0, 0}};
      se::key_t allocation_list[num_voxel_blocks];
      for (size_t i = 0; i < num_voxel_blocks; ++i) {
        allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z());
      }
      octree_.allocate(allocation_list, num_voxel_blocks);

      // Set the values of some voxels.
      constexpr size_t num_voxels = 11;
      const Eigen::Vector3i voxels_coord[num_voxels] =
          {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1},
           {0, 1, 1}, {1, 0, 1}, {1, 1, 0}, {2, 1, 1}, {1, 2, 1},
           {1, 1, 2}};
      for (size_t i = 0; i< num_voxels; ++i) {
        // The values start from value_increment_ and increase by
        // value_increment_ for each next voxel.
        octree_.set(voxels_coord[i].x(), voxels_coord[i].y(), voxels_coord[i].z(),
            value_increment_ * (i + 1));
      }
    }

    typedef se::Octree<TestVoxelT> OctreeF;
    OctreeF octree_;
    static constexpr float value_increment_ = 0.05f;
};



// Get the face neighbor values of a voxel located on the interior of a voxel
// block.
TEST_F(NeighborGatherTest, GetFaceNeighborsLocal) {
  // Safe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_safe
      = octree_.get_face_neighbors<true>(1, 1, 1);
  // Unsafe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_unsafe
      = octree_.get_face_neighbors<false>(1, 1, 1);

  // Voxel -z (1, 1, 0).
  EXPECT_EQ(neighbor_data_safe[0], 8 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[0], 8 * value_increment_);

  // Voxel -y (1, 0, 1).
  EXPECT_EQ(neighbor_data_safe[1], 7 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[1], 7 * value_increment_);

  // Voxel -x (0, 1, 1).
  EXPECT_EQ(neighbor_data_safe[2], 6 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[2], 6 * value_increment_);

  // Voxel +x (2, 1, 1).
  EXPECT_EQ(neighbor_data_safe[3], 9 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[3], 9 * value_increment_);

  // Voxel +y (1, 2, 1).
  EXPECT_EQ(neighbor_data_safe[4], 10 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[4], 10 * value_increment_);

  // Voxel +z (1, 1, 2).
  EXPECT_EQ(neighbor_data_safe[5], 11 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[5], 11 * value_increment_);
}



// Get the face neighbor values of a voxel located on the corner of the volume.
// NOTE Neighbors outside the volume have a value of invalid().
TEST_F(NeighborGatherTest, GetFaceNeighborsVolumeCorner) {
  // Safe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_safe
      = octree_.get_face_neighbors<true>(0, 0, 0);
  // Unsafe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_unsafe
      = octree_.get_face_neighbors<false>(0, 0, 0);

  // Voxel -z (0, 0, -1), outside.
  EXPECT_EQ(neighbor_data_safe[0], TestVoxelT::invalid());

  // Voxel -y (0, -1, 0), outside.
  EXPECT_EQ(neighbor_data_safe[1], TestVoxelT::invalid());

  // Voxel -x (-1, 0, 0), outside.
  EXPECT_EQ(neighbor_data_safe[2], TestVoxelT::invalid());

  // Voxel +x (1, 0, 0), inside.
  EXPECT_EQ(neighbor_data_safe[3], 2 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[3], 2 * value_increment_);

  // Voxel +y (0, 1, 0), inside.
  EXPECT_EQ(neighbor_data_safe[4], 3 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[4], 3 * value_increment_);

  // Voxel +z (0, 0, 1), inside.
  EXPECT_EQ(neighbor_data_safe[5], 4 * value_increment_);
  EXPECT_EQ(neighbor_data_unsafe[5], 4 * value_increment_);
}



// Get the face neighbor values of a voxel located on the corner of a
// VoxelBlock and surrounded by unallocated VoxelBlocks.
// NOTE Unallocated neighbors have a value of initData().
TEST_F(NeighborGatherTest, GetFaceNeighborsCornerUnallocated) {
  // Safe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_safe
      = octree_.get_face_neighbors<true>(octree_.block_size - 1,
      octree_.block_size - 1, octree_.block_size - 1);
  // Unsafe version.
  std::array<TestVoxelT::VoxelData, 6> neighbor_data_unsafe
      = octree_.get_face_neighbors<false>(octree_.block_size - 1,
      octree_.block_size - 1, octree_.block_size - 1);

  // Voxel -z (0, 0, -1), allocated.
  EXPECT_EQ(neighbor_data_safe[0], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[0], TestVoxelT::initData());

  // Voxel -y (0, -1, 0), allocated.
  EXPECT_EQ(neighbor_data_safe[1], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[1], TestVoxelT::initData());

  // Voxel -x (-1, 0, 0), allocated.
  EXPECT_EQ(neighbor_data_safe[2], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[2], TestVoxelT::initData());

  // Voxel +x (1, 0, 0), unallocated.
  EXPECT_EQ(neighbor_data_safe[3], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[3], TestVoxelT::initData());

  // Voxel +y (0, 1, 0), unallocated.
  EXPECT_EQ(neighbor_data_safe[4], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[4], TestVoxelT::initData());

  // Voxel +z (0, 0, 1), unallocated.
  EXPECT_EQ(neighbor_data_safe[5], TestVoxelT::initData());
  EXPECT_EQ(neighbor_data_unsafe[5], TestVoxelT::initData());
}

