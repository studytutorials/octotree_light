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
#include <cmath>
#include "octree.hpp"
#include "utils/math_utils.h"
#include "gtest/gtest.h"
#include "functors/axis_aligned_functor.hpp"
#include "io/ply_io.hpp"
#include "io/vtk-io.h"
#include "algorithms/balancing.hpp"
#include "interpolation/idw_interpolation.hpp"

struct TestVoxelT {
  typedef float VoxelData;
  static inline VoxelData invalid(){ return 0.f; }
  static inline VoxelData initData(){ return 1.f; }

  template <typename T>
  using MemoryPoolType = se::PagedMemoryPool<T>;
  template <typename BufferT>
  using MemoryBufferType = se::PagedMemoryBuffer<BufferT>;
};

float test_fun(float x, float y, float z) {
  return se::math::sq(z) + std::sin(2 * x + y);
}

float sphere_dist(const Eigen::Vector3f& p, const Eigen::Vector3f& C,
    const float radius) {
  const Eigen::Vector3f dir = (C - p).normalized();
  const Eigen::Vector3f centre_offset = p - C;

  const float a = dir.dot(dir);
  const float b = 2 * dir.dot(centre_offset);
  const float c = centre_offset.dot(centre_offset) - radius * radius;
  const float delta = b * b - 4 * a * c;
  float dist = std::numeric_limits<int>::max();
  if(delta > 0) {
    dist = std::min(-b + sqrtf(delta), -b - sqrtf(delta));
    dist /= 2 * a;
  }
  return dist;
}

class InterpolationTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      unsigned size = 256;
      float dim = 5.f;
      octree_.init(size, dim); // 5 meters

      const unsigned centre = size >> 1;
      const unsigned radius = size >> 2;
      const Eigen::Vector3f C(centre, centre, centre);

      for(int z = centre - radius; z < (centre + radius); ++z) {
        for(int y = centre - radius; y < (centre + radius); ++y) {
          for(int x = centre - radius; x < (centre + radius); ++x) {
            const Eigen::Vector3i voxel_coord(x, y, z);
            const float dist = fabs(sphere_dist(voxel_coord.cast<float>(), C, radius));
            if(dist > 20.f && dist < 25.f) {
              allocation_list.push_back(octree_.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
            }
          }
        }
      }
      octree_.allocate(allocation_list.data(), allocation_list.size());

      auto circle_dist = [C, radius](auto& handler, const Eigen::Vector3i& voxel_coord) {
        float data = sphere_dist(voxel_coord.cast<float>(), C, radius);
        handler.set(data);
      };
      se::functor::axis_aligned_map(octree_, circle_dist);

      se::print_octree("./test-sphere.ply", octree_);
      {
        std::stringstream f;
        f << "./sphere-interp.vtk";
        save3DSlice(octree_, Eigen::Vector3i(0, octree_.size()/2, 0),
            Eigen::Vector3i(octree_.size(), octree_.size()/2 + 1, octree_.size()),
            [](const float& data) { return data; }, octree_.maxBlockScale(), f.str().c_str());
      }

      // balance and print.
      se::balance(octree_);
      se::functor::axis_aligned_map(octree_, circle_dist);
      se::print_octree("./test-sphere-balanced.ply", octree_);
      {
        std::stringstream f;
        f << "./sphere-interp-balanced.vtk";
        save3DSlice(octree_, Eigen::Vector3i(0, octree_.size()/2, 0),
            Eigen::Vector3i(octree_.size(), octree_.size()/2 + 1, octree_.size()),
            [](const float& data) { return data; }, octree_.maxBlockScale(), f.str().c_str());
      }

    }

  typedef se::Octree<TestVoxelT> OctreeF;
  OctreeF octree_;
  std::vector<se::key_t> allocation_list;
};

TEST_F(InterpolationTest, IDWInterp) {
  Eigen::Vector3f voxel_coord_f(128.4f, 129.1f, 127.5f);
  auto select_value =  [](const TestVoxelT::VoxelData& data) { return data; };
  se::internal::idw_interp<TestVoxelT::VoxelData>(octree_, voxel_coord_f, select_value);

}

// TEST_F(InterpolationTest, InterpAtPoints) {
//
//   auto test = [this](auto& handler, const Eigen::Vector3i& voxel_coord) {
//     auto data = handler.get();
//     TestVoxelT::VoxelData interpolated = octree_.interp(make_float3(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()), [](const auto& data){ return val.x(); });
//     ASSERT_EQ(data.x(), interpolated);
//   };
//
//   se::functor::axis_aligned<TestVoxelT, Octree, decltype(test)>
//     funct_test(octree_, test);
//   funct_test.apply();
// }
