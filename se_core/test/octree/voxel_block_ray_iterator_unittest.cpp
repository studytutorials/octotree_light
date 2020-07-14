/*
 * Copyright 2016 Emanuele Vespa, Imperial College London
 * Copyright 2020 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <vector>

#include <gtest/gtest.h>

#include <se/octree.hpp>
#include <se/voxel_block_ray_iterator.hpp>



// Create a voxel trait storing a single float value.
struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return -1.f; }
  static inline VoxelData initData(){ return 0.f; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



class VoxelBlockRayIteratorTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(res_, dim_);

      ray_origin_M_   = Eigen::Vector3f::Constant(1.5f);
      ray_dir_M_ = Eigen::Vector3f::Ones().normalized();

      // Ensure stepsize is big enough to get distinct blocks
      const float stepsize = 2 * voxel_dim_ * se::Octree<TestVoxelT>::block_size;

      // Allocate voxel blocks hit by the ray defined by ray_origin_M_ and ray_dir_M_
      const int num_blocks = 4;
      float t = 0.6f;
      for (int i = 0; i < num_blocks; ++i, t += stepsize) {
        const Eigen::Vector3f ray_pos_M = ray_origin_M_ + t * ray_dir_M_;
        const Eigen::Vector3i voxel_coord = (ray_pos_M / voxel_dim_).cast<int>();

        // Hash to VoxelBlocks
        const se::key_t key = octree_.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
        allocation_list_.push_back(key);
      }
      octree_.allocate(allocation_list_.data(), allocation_list_.size());
    }

    se::Octree<TestVoxelT> octree_;
    const int res_ = 512;
    const float dim_ = 5.f;
    const float voxel_dim_ = dim_ / res_;
    Eigen::Vector3f ray_origin_M_;
    Eigen::Vector3f ray_dir_M_;
    std::vector<se::key_t> allocation_list_;
};



TEST_F(VoxelBlockRayIteratorTest, FetchAlongRay) {
  se::VoxelBlockRayIterator<TestVoxelT> it (octree_, ray_origin_M_, ray_dir_M_,
      0.4, 4.0f);
  size_t i = 0;
  TestVoxelT::VoxelBlockType* current;
  while ((current = it.next())) {
    ASSERT_LT(i, allocation_list_.size());
    ASSERT_EQ(current->code(), allocation_list_[i]);
    i++;
  }
  ASSERT_EQ(i, allocation_list_.size());
}

