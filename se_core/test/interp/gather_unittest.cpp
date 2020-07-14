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

#include <gtest/gtest.h>

#include <se/interpolation/interp_gather.hpp>
#include <se/node_iterator.hpp>
#include <se/octree.hpp>
#include <se/utils/math_utils.h>

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

class GatherTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(512, 5);
      const Eigen::Vector3i blocks_coord[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
      {136, 128, 128}, {128, 136, 128}, {136, 136, 128},
      {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
      se::key_t allocation_list[10];
      for(int i = 0; i < 10; ++i) {
        allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z());
      }
      octree_.allocate(allocation_list, 10);
    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
};

TEST_F(GatherTest, Init) {
  EXPECT_EQ(octree_.get(137, 138, 130), TestVoxelT::initData());
}

TEST_F(GatherTest, GatherLocal) {
  TestVoxelT::VoxelData data[8];
  const Eigen::Vector3i base_coord = {136, 128, 136};
  se::internal::gather_values(octree_, base_coord, 1, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, ZCrosses) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {132, 128, 135};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            (base_coord.z() % block_size == block_size - 1);
  ASSERT_EQ(crossmask, 1u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, YCrosses) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {132, 135, 132};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 2u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XCrosses) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {135, 132, 132};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 4u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, YZCross) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {129, 135, 135};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 3u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XZCross) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {135, 131, 135};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 5u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, XYCross) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {135, 135, 138};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 6u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, AllCross) {
  TestVoxelT::VoxelData data[8];
  const unsigned block_size = TestVoxelT::VoxelBlockType::size_li;
  const Eigen::Vector3i base_coord = {135, 135, 135};
  unsigned int crossmask = ((base_coord.x() % block_size == block_size - 1) << 2) |
                           ((base_coord.y() % block_size == block_size - 1) << 1) |
                            ((base_coord.z() % block_size) == block_size - 1);
  ASSERT_EQ(crossmask, 7u);
  se::internal::gather_values(octree_, base_coord, 0, [](const auto& data){ return data; }, data);

  for(int i = 0; i < 8; ++i) {
    EXPECT_EQ(data[i], TestVoxelT::initData());
  }
}

TEST_F(GatherTest, FetchWithStack) {
  se::Node<TestVoxelT>* stack[CAST_STACK_DEPTH] = {nullptr};
  int voxel_depth = octree_.voxelDepth();
  se::internal::fetch(stack, octree_.root(), voxel_depth,
      Eigen::Vector3i(128, 136, 128));
  int l = 0;
  while(stack[l] && stack[l + 1] != nullptr) {
    ASSERT_TRUE(se::parent(stack[l + 1]->code(), voxel_depth) == stack[l]->code());
    ++l;
  }
}

TEST_F(GatherTest, FetchNeighbourhood) {
  Eigen::Vector3i base_coord(64,   64, 0);
  Eigen::Vector3i node_0_coord(128,   0, 0); // (1, 0, 0)
  Eigen::Vector3i node_1_coord(0,   128, 0); // (0, 1, 0)
  Eigen::Vector3i node_2_coord(128, 128, 0); // (1, 1, 0)
  octree_.insert(base_coord.x(), base_coord.y(), base_coord.z(), 3);
  octree_.insert(node_0_coord.x(), node_0_coord.y(), node_0_coord.z(), 2);
  octree_.insert(node_1_coord.x(), node_1_coord.y(), node_1_coord.z(), 2);
  octree_.insert(node_2_coord.x(), node_2_coord.y(), node_2_coord.z(), 2);

  se::Node<TestVoxelT>* stack[CAST_STACK_DEPTH] = {nullptr};
  int voxel_depth = octree_.voxelDepth();
  auto base = se::internal::fetch(stack, octree_.root(), voxel_depth, base_coord);
  auto node_1 = se::internal::fetch_neighbour(stack, base, voxel_depth, 1);
  auto node_2 = se::internal::fetch_neighbour(stack, base, voxel_depth, 2);
  auto node_3 = se::internal::fetch_neighbour(stack, base, voxel_depth, 3);
  ASSERT_TRUE(se::keyops::decode(base->code()) == base_coord);
  ASSERT_TRUE(se::keyops::decode(node_1->code()) == node_0_coord);
  ASSERT_TRUE(se::keyops::decode(node_2->code()) == node_1_coord);
  ASSERT_TRUE(se::keyops::decode(node_3->code()) == node_2_coord);

  std::fill(std::begin(stack), std::end(stack), nullptr);
  base = se::internal::fetch(stack, octree_.root(), voxel_depth, node_0_coord);
  auto failed = se::internal::fetch_neighbour(stack, base, voxel_depth, 1);
  ASSERT_TRUE(failed == nullptr);
}
