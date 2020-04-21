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
#include "utils/math_utils.h"
#include "geometry/octree_collision.hpp"
#include "geometry/aabb_collision.hpp"
#include "utils/morton_utils.hpp"
#include "octree.hpp"
#include "octant_ops.hpp"
#include "node_iterator.hpp"
#include "functors/axis_aligned_functor.hpp"
#include "gtest/gtest.h"

using namespace se::geometry;

struct TestVoxelT{
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

collision_status test_voxel(const TestVoxelT::VoxelData& data) {
  if(data == TestVoxelT::initData()) return collision_status::unseen;
  if(data == 10.f) return collision_status::empty;
  return collision_status::occupied;
};

class OctreeCollisionTest : public ::testing::Test {
  protected:
    virtual void SetUp() {

      octree_.init(256, 5);
      const Eigen::Vector3i blocks_coord[1] = {{56, 12, 254}};
      se::key_t allocation_list[1];
      allocation_list[0] = octree_.hash(blocks_coord[0].x(), blocks_coord[0].y(), blocks_coord[0].z());
      octree_.allocate(allocation_list, 1);

      auto set_to_ten = [](auto& handler, const Eigen::Vector3i& coord) {
        if((coord.array() >= Eigen::Vector3i(48, 0, 240).array()).all()){
          handler.set(10.f);
        }
      };
      se::functor::axis_aligned_map(octree_, set_to_ten);
    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
};

TEST_F(OctreeCollisionTest, TotallyUnseen) {

  se::node_iterator<TestVoxelT> it(octree_);
  se::Node<TestVoxelT>* node = it.next();
  for(int i = 256; node != nullptr ; node = it.next(), i /= 2){
    const Eigen::Vector3i node_coord = se::keyops::decode(node->code_);
    const int node_size = node->size_;
    const se::Octree<TestVoxelT>::VoxelData data = (node->data_[0]);
    printf("se::Node's coordinates: (%d, %d, %d), size %d, value %.2f\n",
        node_coord.x(), node_coord.y(), node_coord.z(), node_size, data);
    EXPECT_EQ(node_size, i);
  }

  const Eigen::Vector3i bbox_coord = {23, 0, 100};
  const Eigen::Vector3i bbox_size = {2, 2, 2};

  const collision_status collides = collides_with(octree_, bbox_coord, bbox_size,
      test_voxel);
  ASSERT_EQ(collides, collision_status::unseen);
}

TEST_F(OctreeCollisionTest, PartiallyUnseen) {
  const Eigen::Vector3i bbox_coord = {47, 0, 239};
  const Eigen::Vector3i bbox_size = {6, 6, 6};
  const collision_status collides = collides_with(octree_, bbox_coord, bbox_size,
      test_voxel);
  ASSERT_EQ(collides, collision_status::unseen);
}

TEST_F(OctreeCollisionTest, Empty) {
  const Eigen::Vector3i bbox_coord = {49, 1, 242};
  const Eigen::Vector3i bbox_size = {1, 1, 1};
  const collision_status collides = collides_with(octree_, bbox_coord, bbox_size,
      test_voxel);
  ASSERT_EQ(collides, collision_status::empty);
}

TEST_F(OctreeCollisionTest, Collision){
  const Eigen::Vector3i bbox_coord = {54, 10, 249};
  const Eigen::Vector3i bbox_size = {5, 5, 3};

  auto update = [](auto& handler, const Eigen::Vector3i& coord) {
      handler.set(2.f);
  };
  se::functor::axis_aligned_map(octree_, update);

  const collision_status collides = collides_with(octree_, bbox_coord, bbox_size,
      test_voxel);
  ASSERT_EQ(collides, collision_status::occupied);
}

TEST_F(OctreeCollisionTest, CollisionFreeLeaf){
  // Allocated block: {56, 8, 248};
  const Eigen::Vector3i bbox_coord = {61, 13, 253};
  const Eigen::Vector3i bbox_size = {2, 2, 2};

  /* Update blocks_coord as occupied node */
  se::VoxelBlock<TestVoxelT>* block = octree_.fetch(56, 12, 254);
  const Eigen::Vector3i block_coord = block->coordinates();
  int x, y, z, block_size;
  block_size = (int) se::VoxelBlock<TestVoxelT>::size;
  int x_last = block_coord.x() + block_size;
  int y_last = block_coord.y() + block_size;
  int z_last = block_coord.z() + block_size;
  for(z = block_coord.z(); z < z_last; ++z){
    for (y = block_coord.y(); y < y_last; ++y){
      for (x = block_coord.x(); x < x_last; ++x){
        if(x < x_last / 2 && y < y_last / 2 && z < z_last / 2)
          block->data(Eigen::Vector3i(x, y, z), 2.f);
        else
          block->data(Eigen::Vector3i(x, y, z), 10.f);

      }
    }
  }

  const collision_status collides = collides_with(octree_, bbox_coord, bbox_size,
      test_voxel);
  ASSERT_EQ(collides, collision_status::empty);
}
