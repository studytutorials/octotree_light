#include "se/octant_ops.hpp"
#include "se/octree.hpp"
#include "se/algorithms/balancing.hpp"
#include "se/functors/axis_aligned_functor.hpp"
#include "se/functors/for_each.hpp"
#include "se/io/vtk-io.h"
#include "se/io/ply_io.hpp"
#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"
#include "../../src/MultiresTSDF/MultiresTSDF.cpp"
#include "../../src/MultiresTSDF/MultiresTSDF_allocation.cpp"
#include "../../src/MultiresTSDF/MultiresTSDF_mapping.cpp"
#include "../../src/MultiresTSDF/MultiresTSDF_rendering.cpp"
#include <random>
#include <functional>
#include <gtest/gtest.h>

float sphereDist(const Eigen::Vector3f& point,
                 const Eigen::Vector3f& centre,
                 const float            radius) {
  const float dist = (point - centre).norm();
  return dist - radius;
}

float sphereDistNoisy(const Eigen::Vector3f& point,
                      const Eigen::Vector3f& centre,
                      const float            radius) {
  static std::mt19937 gen{1};
  std::normal_distribution<> noise(0, 2);
  const float dist = (point - centre).norm();
  return (dist - radius) + noise(gen);
}

template <typename T>
void updateBlock(se::VoxelBlock<T>*     block,
                 const Eigen::Vector3f& centre,
                 const float            radius,
                 const int              scale) {
  const Eigen::Vector3i block_coord = block->coordinates();
  const int block_size = se::VoxelBlock<T>::size;
  const int stride = 1 << scale;
  for(int z = 0; z < block_size; z += stride) {
    for (int y = 0; y < block_size; y += stride) {
      for (int x = 0; x < block_size; x += stride) {
        const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
        auto voxel_data = block->data(voxel_coord, scale);
        const float tsdf_value = sphereDistNoisy(
            voxel_coord.cast<float>() + float(stride) * Eigen::Vector3f::Constant(0.5f),
            centre, radius);
        voxel_data.delta_y++;
        voxel_data.x = (voxel_data.x * voxel_data.y + tsdf_value) / (voxel_data.y + 1);
        voxel_data.y = voxel_data.y + 1;
        block->data(voxel_coord, scale, voxel_data);
      }
    }
  }
}

class MultiresESDFMovingSphereTest : public ::testing::Test {
  protected:
    virtual void SetUp() {
      unsigned size = 256;
      float dim = 5.f;
      octree_.init(size, dim); // 5 meters

      centre_ = size >> 1;
      radius_ = size >> 2;
      const Eigen::Vector3f C(centre_, centre_, centre_);

      for(int z = centre_ - radius_; z < (centre_ + radius_); ++z) {
        for(int y = centre_ - radius_; y < (centre_ + radius_); ++y) {
          for(int x = centre_ - radius_; x < (centre_ + radius_); ++x) {
            const Eigen::Vector3i voxel_coord(x, y, z);
            const float dist = fabs(sphereDist(voxel_coord.cast<float>(), C, radius_));
            if(dist > 20.f && dist < 25.f) {
              allocation_list.push_back(octree_.hash(voxel_coord.x(), voxel_coord.y(), voxel_coord.z()));
            }
          }
        }
      }
      octree_.allocate(allocation_list.data(), allocation_list.size());
    }

  typedef se::Octree<MultiresTSDF::VoxelType> OctreeT;
  OctreeT octree_;
  int centre_;
  int radius_;
  std::vector<se::key_t> allocation_list;
};

TEST_F(MultiresESDFMovingSphereTest, Integration) {
  Eigen::Vector3f centre = Eigen::Vector3f::Constant(centre_);
  int scale = 0;
  float radius = this->radius_;
  auto update_op = [&centre, &scale, radius](se::VoxelBlock<MultiresTSDF::VoxelType>* block) {
    updateBlock(block, centre, radius, scale);
  };

  for(int i = 0; i < 5; ++i) {
    se::functor::internal::parallel_for_each(octree_.pool().blockBuffer(), update_op);
    auto op = [](se::VoxelBlock<MultiresTSDF::VoxelType>* block) { se::multires::propagateUp(block, 0); };
    se::functor::internal::parallel_for_each(octree_.pool().blockBuffer(), op);

    {
      std::stringstream f;
      f << "./out/sphere-interp-" << i << ".vtk";
      save3DSlice(octree_, Eigen::Vector3i(0, octree_.size() / 2, 0),
          Eigen::Vector3i(octree_.size(), octree_.size() / 2 + 1, octree_.size()),
          [](const auto& data) { return data.x; }, octree_.maxBlockScale(), f.str().c_str());
    }
  }

  // update captured lambda parameters
  centre += Eigen::Vector3f::Constant(10.f);
  scale = 2;
  for(int i = 5; i < 10; ++i) {
    se::functor::internal::parallel_for_each(octree_.pool().blockBuffer(), update_op);
    auto& octree_ref = octree_;
    auto op = [&octree_ref, scale](se::VoxelBlock<MultiresTSDF::VoxelType>* block) { se::multires::propagateDown(octree_ref, block, scale, 0); };
    se::functor::internal::parallel_for_each(octree_.pool().blockBuffer(), op);

    {
      std::stringstream f;
      f << "./out/sphere-interp-" << i << ".vtk";
      save3DSlice(octree_, Eigen::Vector3i(0, octree_.size() / 2, 0),
          Eigen::Vector3i(octree_.size(), octree_.size() / 2 + 1, octree_.size()),
          [](const auto& data) { return data.x; }, octree_.maxBlockScale(), f.str().c_str());
    }
  }
  se::print_octree("./out/test-sphere.ply", octree_);
}
