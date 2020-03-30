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

#include "octree.hpp"
#include "voxel_block_ray_iterator.hpp"
#include "gtest/gtest.h"
#include <vector>



// Create a voxel trait storing a single float value.
struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData empty(){ return -1.f; }
  static inline VoxelData initValue(){ return 0.f; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};



class VoxelBlockRayIteratorTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(res_, dim_);

      origin_   = Eigen::Vector3f::Constant(1.5f);
      direction_ = Eigen::Vector3f::Ones().normalized();

      // Ensure stepsize is big enough to get distinct blocks
      const float stepsize = 2 * voxel_size_ * se::Octree<TestVoxelT>::blockSide;

      // Allocate voxel blocks hit by the ray defined by origin_ and direction_
      const int num_blocks = 4;
      float t = 0.6f;
      for (int i = 0; i < num_blocks; ++i, t += stepsize) {
        const Eigen::Vector3f point = origin_ + t * direction_;
        const Eigen::Vector3i voxel = (point / voxel_size_).cast<int>();

        // Hash to VoxelBlocks
        const se::key_t key = octree_.hash(voxel.x(), voxel.y(), voxel.z());
        alloc_list_.push_back(key);
      }
      octree_.allocate(alloc_list_.data(), alloc_list_.size());
    }

    se::Octree<TestVoxelT> octree_;
    const int res_ = 512;
    const float dim_ = 5.f;
    const float voxel_size_ = dim_ / res_;
    Eigen::Vector3f origin_;
    Eigen::Vector3f direction_;
    std::vector<se::key_t> alloc_list_;
};



TEST_F(VoxelBlockRayIteratorTest, FetchAlongRay) {
  se::VoxelBlockRayIterator<TestVoxelT> it (octree_, origin_, direction_,
      0.4, 4.0f);
  int i = 0;
  se::VoxelBlock<TestVoxelT>* current;
  while (current = it.next()) {
    ASSERT_LT(i, alloc_list_.size());
    ASSERT_EQ(current->code_, alloc_list_[i]);
    i++;
  }
  ASSERT_EQ(i, alloc_list_.size());
}

