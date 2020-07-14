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

#include <random>

#include <gtest/gtest.h>

#include <se/node_iterator.hpp>
#include <se/octant_ops.hpp>
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

class MultiscaleTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      octree_.init(512, 5);

    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
};

TEST_F(MultiscaleTest, Init) {
  EXPECT_EQ(octree_.get(137, 138, 130), TestVoxelT::initData());
}

TEST_F(MultiscaleTest, PlainAlloc) {
  const Eigen::Vector3i blocks_coord[2] = {{56, 12, 254}, {87, 32, 423}};
  se::key_t allocation_list[2];
  for(int i = 0; i < 2; ++i) {
    allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z());
  }
  octree_.allocate(allocation_list, 2);

  octree_.set(56, 12, 254, 3.f);

  EXPECT_EQ(octree_.get(56, 12, 254), 3.f);
  EXPECT_EQ(octree_.get(106, 12, 254), TestVoxelT::initData());
  EXPECT_NE(octree_.get(106, 12, 254), 3.f);
}

TEST_F(MultiscaleTest, ScaledAlloc) {
  const Eigen::Vector3i blocks_coord[2] = {{200, 12, 25}, {87, 32, 423}};
  se::key_t allocation_list[2];
  for(int i = 0; i < 2; ++i) {
    allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z(), 5);
  }

  octree_.allocate(allocation_list, 2);
  se::Node<TestVoxelT>* node = octree_.fetch_node(87, 32, 420, 5);
  ASSERT_TRUE(node != NULL);
  node->childData(0, 10.f);
  EXPECT_EQ(octree_.get(87, 32, 420), 10.f);
}

TEST_F(MultiscaleTest, Iterator) {
  const Eigen::Vector3i blocks_coord[1] = {{56, 12, 254}};
  se::key_t allocation_list[1];
  allocation_list[0] = octree_.hash(blocks_coord[0].x(), blocks_coord[0].y(), blocks_coord[0].z());

  octree_.allocate(allocation_list, 1);
  se::node_iterator<TestVoxelT> it(octree_);
  se::Node<TestVoxelT>* node = it.next();
  for(int i = 512; node != nullptr; node = it.next(), i /= 2){
    const int node_size = node->size();
    EXPECT_EQ(node_size, i);
  }
}

TEST_F(MultiscaleTest, ChildrenMaskTest) {
  const Eigen::Vector3i blocks_coord[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
    {136, 128, 128}, {128, 136, 128}, {136, 136, 128},
    {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
  se::key_t allocation_list[10];
  for(int i = 0; i < 10; ++i) {
    allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z(), 5);
  }

  octree_.allocate(allocation_list, 10);
  const se::PagedMemoryBuffer<se::Node<TestVoxelT> >& nodes_buffer = octree_.pool().nodeBuffer();
  const size_t num_nodes = nodes_buffer.size();
  for(size_t i = 0; i < num_nodes; ++i) {
    se::Node<TestVoxelT>* node = nodes_buffer[i];
    for(int child_idx = 0; child_idx < 8; ++child_idx) {
      if(node->child(child_idx)) {
        ASSERT_TRUE(node->children_mask() & (1 << child_idx));
      }
    }
  }
}

TEST_F(MultiscaleTest, OctantAlloc) {
  const Eigen::Vector3i blocks_coord[10] = {{56, 12, 254}, {87, 32, 423}, {128, 128, 128},
    {136, 128, 128}, {128, 136, 128}, {136, 136, 128},
    {128, 128, 136}, {136, 128, 136}, {128, 136, 136}, {136, 136, 136}};
  se::key_t allocation_list[10];
  for(int i = 0; i < 10; ++i) {
    allocation_list[i] = octree_.hash(blocks_coord[i].x(), blocks_coord[i].y(), blocks_coord[i].z());
  }

  allocation_list[2] = allocation_list[2] | 3;
  allocation_list[9] = allocation_list[2] | 5;
  octree_.allocate(allocation_list, 10);
  se::Node<TestVoxelT>* node = octree_.fetch_node(blocks_coord[4].x(), blocks_coord[4].y(),
      blocks_coord[4].z(), 3);
  ASSERT_TRUE(node != nullptr);
  node = octree_.fetch_node(blocks_coord[9].x(), blocks_coord[9].y(),
      blocks_coord[9].z(), 6);
  ASSERT_TRUE(node == nullptr);
}

TEST_F(MultiscaleTest, SingleInsert) {
  Eigen::Vector3i voxel_coord(32, 208, 44);
  const int block_size = TestVoxelT::VoxelBlockType::size_li;
  TestVoxelT::VoxelBlockType* block = octree_.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
  Eigen::Vector3i block_coord = block->coordinates();
  Eigen::Vector3i block_coord_rounded = block_size * (voxel_coord / block_size);
  ASSERT_TRUE(block_coord == block_coord_rounded);
}

TEST_F(MultiscaleTest, MultipleInsert) {
  OctreeF octree;
  octree.init(1024, 10);
  const int block_depth = octree.blockDepth();
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(0, 1023);

  int num_tested = 0;
  for(int i = 1, node_size = octree.size() / 2; i <= block_depth; ++i, node_size = node_size / 2) {
    for(int j = 0; j < 20; ++j) {
      Eigen::Vector3i voxel_coord(dis(gen), dis(gen), dis(gen));
      octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), i);
      se::Node<TestVoxelT>* fetched_node = octree.fetch_node(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), i);
      Eigen::Vector3i node_coord = se::keyops::decode(fetched_node->code());
      Eigen::Vector3i node_coord_rounded = node_size * (voxel_coord / node_size);

      // Check expected coordinates
      ASSERT_TRUE(node_coord == node_coord_rounded);
      // Should not have any children up to this depth
      ASSERT_TRUE(fetched_node->children_mask() == 0);
      ++num_tested;
    }
  }
}

struct TestVoxel2T {
  typedef Eigen::Vector3i VoxelData;
  static inline VoxelData invalid(){ return Eigen::Vector3i::Zero(); }
  static inline VoxelData initData(){ return Eigen::Vector3i::Zero(); }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxel2T>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxel2T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

TEST(MultiscaleBlock, ReadWrite) {
  se::Octree<TestVoxel2T> octree;
  octree.init(1024, 10);
  const int block_size = TestVoxel2T::VoxelBlockType::size_li;
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(1); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(0, 1023);
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i>> voxels_coord;

  const int n = 50;
  for(int j = 0; j < n; ++j) {
    voxels_coord.push_back(Eigen::Vector3i(dis(gen), dis(gen), dis(gen)));
    const Eigen::Vector3i& voxel_coord = voxels_coord.back();
    auto* block = octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    const Eigen::Vector3i& block_coord = block->coordinates();
    for(int z = 0; z < block_size; ++z) {
      for(int y = 0; y < block_size; ++y) {
        for(int x = 0; x < block_size; ++x) {
          const Eigen::Vector3i voxel_coord_scale_0 = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3i voxel_coord_scale_1 = (voxel_coord_scale_0 / 2) * 2;
          const Eigen::Vector3i voxel_coord_scale_2 = (voxel_coord_scale_0 / 4) * 4;
          block->setData(voxel_coord_scale_0, voxel_coord_scale_0);
          block->setData(voxel_coord_scale_0, 1, voxel_coord_scale_1);
          block->setData(voxel_coord_scale_0, 2, voxel_coord_scale_2);
        }
      }
    }
  }

  for(size_t i = 0; i < voxels_coord.size(); ++i) {
    const Eigen::Vector3i& voxel_coord = voxels_coord[i];
    auto* block = octree.fetch(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
    const Eigen::Vector3i& block_coord = block->coordinates();
    for(int z = 0; z < block_size; ++z) {
      for(int y = 0; y < block_size; ++y) {
        for(int x = 0; x < block_size; ++x) {
          const Eigen::Vector3i voxel_coord_scale_0 = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3i voxel_coord_scale_1 = (voxel_coord_scale_0 / 2) * 2;
          const Eigen::Vector3i voxel_coord_scale_2 = (voxel_coord_scale_0 / 4) * 4;
          ASSERT_TRUE(block->data(voxel_coord_scale_0).cwiseEqual(voxel_coord_scale_0).all());
          ASSERT_TRUE(block->data(voxel_coord_scale_0, 1).cwiseEqual(voxel_coord_scale_1).all());
          ASSERT_TRUE(block->data(voxel_coord_scale_0, 2).cwiseEqual(voxel_coord_scale_2).all());
        }
      }
    }
  }
}
