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

#include <se/functors/axis_aligned_functor.hpp>
#include <se/octree.hpp>
#include <se/utils/math_utils.h>

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 0.f; }

  using VoxelBlockType = se::VoxelBlockFull<TestVoxelT>;

  using MemoryPoolType = se::PagedMemoryPool<TestVoxelT>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

class AxisAlignedTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      unsigned size = 256;
      float dim = 5.f;
      octree_.init(size, dim); // 5 meters

      const float voxel_dim = octree_.dim() / octree_.size();
      const float inverse_voxel_dim = 1.f / voxel_dim;
      const int band = 1 * inverse_voxel_dim;
      const Eigen::Vector3i offset =
        Eigen::Vector3i::Constant(octree_.size() / 2 - band / 2);
      for(int z = 0; z < band; ++z) {
        for(int y = 0; y < band; ++y) {
          for(int x = 0; x < band; ++x) {
            const Eigen::Vector3i voxel_coord =  Eigen::Vector3i(x + offset.x(),
                y + offset(1), z + offset(2));
            allocation_list.push_back(octree_.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(), octree_.blockDepth()));
          }
        }
      }
      octree_.allocate(allocation_list.data(), allocation_list.size());
    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
  std::vector<se::key_t> allocation_list;
};

TEST_F(AxisAlignedTest, Init) {

  auto initialise = [](auto& handler, const Eigen::Vector3i&) {
    handler.set(TestVoxelT::initData());
  };

  auto test = [](auto& handler, const Eigen::Vector3i&) {
    auto data = handler.get();
    ASSERT_EQ(data, TestVoxelT::initData());
  };

  se::functor::axis_aligned_map(octree_, initialise);
  se::functor::axis_aligned_map(octree_, test);
}

TEST_F(AxisAlignedTest, BBoxTest) {

  auto set_to_ten = [](auto& handler, const Eigen::Vector3i&) {
          handler.set(10.f);
    };

  se::functor::axis_aligned_map(octree_, set_to_ten,
      Eigen::Vector3i::Constant(100), Eigen::Vector3i::Constant(151));

  for(int z = 50; z < 200; ++z)
    for(int y = 50; y < 200; ++y)
      for(int x = 50; x < 200; ++x) {
        auto* block = octree_.fetch(x, y, z);
        if(block &&
           se::math::in(x, 100, 150) &&
           se::math::in(y, 100, 150) &&
           se::math::in(z, 100, 150)){
          ASSERT_EQ(block->data(Eigen::Vector3i(x, y, z)), 10.f);
        }
        else if(block) {
          ASSERT_EQ(block->data(Eigen::Vector3i(x, y, z)),
                    TestVoxelT::initData());
        }
      }
}
