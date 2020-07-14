/*

Copyright 2020 Nils Funk, Imperial College London

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

#include <gtest/gtest.h>

#include <se/algorithms/balancing.hpp>
#include <se/io/octree_io.hpp>
#include <se/octant_ops.hpp>
#include <se/octree.hpp>
#include <se/utils/math_utils.h>

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 0.f; }

  using VoxelBlockType = se::VoxelBlockSingle<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(VoxelBlock, InitData) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  Eigen::Vector3i voxel_coord_2(8, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  octree.insert(voxel_coord_2.x(), voxel_coord_2.y(), voxel_coord_2.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0.f, block_1->initData());

  TestVoxelT::VoxelBlockType* block_2 = octree.fetch(voxel_coord_2.x(), voxel_coord_2.y(), voxel_coord_2.z());
  block_2->setInitData(2.f);
  ASSERT_EQ(2.f, block_2->initData());

  TestVoxelT::VoxelBlockType* block_3 = new TestVoxelT::VoxelBlockType(3.f);
  ASSERT_EQ(3.f, block_3->initData());
  delete(block_3);
}

TEST(VoxelBlock, DataIOEigen) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0u, block_1->blockData().size());

  block_1->allocateDownTo(se::VoxelBlock<TestVoxelT>::max_scale);
  block_1->setData(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale, 1.f);
  ASSERT_EQ(1u, block_1->blockData().size());
  ASSERT_EQ(1.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale - 1));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, 0));

  block_1->allocateDownTo(0);
  block_1->setData(voxel_coord_1, 0, se::VoxelBlock<TestVoxelT>::max_scale);
  ASSERT_EQ(3u + 1u, block_1->blockData().size());
  ASSERT_EQ(3.f, block_1->data(voxel_coord_1, 0));
  ASSERT_EQ(3.f, block_1->data(voxel_coord_1));
}

TEST(VoxelBlock, DataIOIndex) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0u, block_1->blockData().size());

  block_1->allocateDownTo(3);
  block_1->setData(512 + 64 + 8, 1.f);
  ASSERT_EQ(1u, block_1->blockData().size());
  ASSERT_EQ(1.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale - 1));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, 0));

  block_1->allocateDownTo(0);
  block_1->setData(0, 2.f);
  block_1->setData(se::VoxelBlock<TestVoxelT>::size_cu - 1, 3.f);
  ASSERT_EQ(2.f, block_1->data(0));
  ASSERT_EQ(3.f, block_1->data(se::VoxelBlock<TestVoxelT>::size_cu - 1));
  ASSERT_EQ(2.f, block_1->data(voxel_coord_1));
  ASSERT_EQ(2.f, block_1->data(voxel_coord_1, 0));
  ASSERT_EQ(3.f, block_1->data(Eigen::Vector3i(7,7,7)));
  ASSERT_EQ(3.f, block_1->data(Eigen::Vector3i(7,7,7), 0));
}

TEST(VoxelBlock, DataIOSafeEigen) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0u, block_1->blockData().size());

  block_1->setDataSafe(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale, 1.f);
  ASSERT_EQ(1u, block_1->blockData().size());
  ASSERT_EQ(1.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale - 1));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, 0));

  block_1->setDataSafe(voxel_coord_1, 0, se::VoxelBlock<TestVoxelT>::max_scale);
  ASSERT_EQ(3u + 1u, block_1->blockData().size());
  ASSERT_EQ(3.f, block_1->data(voxel_coord_1, 0));
  ASSERT_EQ(3.f, block_1->data(voxel_coord_1));
}

TEST(VoxelBlock, DataIOSafeIndex) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0u, block_1->blockData().size());

  block_1->setDataSafe(512 + 64 + 8, 1.f);
  ASSERT_EQ(1u, block_1->blockData().size());
  ASSERT_EQ(1.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, se::VoxelBlock<TestVoxelT>::max_scale - 1));
  ASSERT_EQ(0.f, block_1->data(voxel_coord_1, 0));

  block_1->setDataSafe(0, 2.f);
  block_1->setDataSafe(se::VoxelBlock<TestVoxelT>::size_cu - 1, 3.f);
  ASSERT_EQ(2.f, block_1->data(0));
  ASSERT_EQ(3.f, block_1->data(se::VoxelBlock<TestVoxelT>::size_cu - 1));
  ASSERT_EQ(2.f, block_1->data(voxel_coord_1));
  ASSERT_EQ(2.f, block_1->data(voxel_coord_1, 0));
  ASSERT_EQ(3.f, block_1->data(Eigen::Vector3i(7,7,7)));
  ASSERT_EQ(3.f, block_1->data(Eigen::Vector3i(7,7,7), 0));
}

TEST(VoxelBlock, DataIODelete) {
  se::Octree<TestVoxelT> octree;
  const unsigned int voxel_depth = 5;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord_1(0, 0, 0);
  octree.insert(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());

  TestVoxelT::VoxelBlockType* block_1 = octree.fetch(voxel_coord_1.x(), voxel_coord_1.y(), voxel_coord_1.z());
  ASSERT_EQ(0u, block_1->blockData().size());
  ASSERT_EQ(-1, block_1->min_scale());

  block_1->setDataSafe(voxel_coord_1, 1.f);
  ASSERT_EQ(4u, block_1->blockData().size());
  ASSERT_EQ(0, block_1->min_scale());

  block_1->deleteUpTo(1);
  ASSERT_EQ(3u, block_1->blockData().size());
  ASSERT_EQ(1, block_1->min_scale());

  block_1->setDataSafe(voxel_coord_1, 1.f);
  ASSERT_EQ(4u, block_1->blockData().size());
  ASSERT_EQ(0, block_1->min_scale());

  block_1->deleteUpTo(3);
  ASSERT_EQ(1u, block_1->blockData().size());
  ASSERT_EQ(3, block_1->min_scale());
}
