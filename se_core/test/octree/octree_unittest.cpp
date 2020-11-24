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

#include <se/algorithms/balancing.hpp>
#include <se/io/octree_io.hpp>
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

TEST(Octree, OctantFaceNeighbours) {
  const Eigen::Vector3i octant_coord = {112, 80, 160};
  const unsigned int voxel_depth = 8;
  const unsigned int block_depth = 5;
  const se::key_t octant_key =
    se::keyops::encode(octant_coord.x(), octant_coord.y(), octant_coord.z(), block_depth, voxel_depth);
  const unsigned int size = 8;
  const Eigen::Vector3i faces[6] = {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0},
    {0, 0, -1}, {0, 0, 1}};
  for(int i = 0; i < 6; ++i) {
    const Eigen::Vector3i neighbour_coord = octant_coord + size * faces[i];
    const Eigen::Vector3i computed_neighbour_coord = se::face_neighbour(octant_key, i, block_depth, voxel_depth);
    ASSERT_EQ(neighbour_coord.x(), computed_neighbour_coord.x());
    ASSERT_EQ(neighbour_coord.y(), computed_neighbour_coord.y());
    ASSERT_EQ(neighbour_coord.z(), computed_neighbour_coord.z());
  }
}

TEST(Octree, OctantDescendant) {
  // First octant
  unsigned voxel_depth = 8;
  unsigned block_depth = voxel_depth - 3;
  se::key_t octant_key = se::keyops::encode(110, 80, 159, block_depth, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, octant_key, voxel_depth));

  se::key_t ancestor_key = se::keyops::encode(96, 64, 128, 3, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(110, 80, 159, 3, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(110, 80, 159, 4, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(128, 64, 64, 3, voxel_depth);
  EXPECT_FALSE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  // Second octant
  voxel_depth = 7;
  block_depth = voxel_depth - 3;
  octant_key = se::keyops::encode(80, 64, 48, block_depth, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(80, 64, 48, block_depth - 1, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(81, 65, 49, block_depth - 1, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));

  ancestor_key = se::keyops::encode(80, 64, 48, block_depth - 2, voxel_depth);
  EXPECT_TRUE(se::descendant(octant_key, ancestor_key, voxel_depth));
  EXPECT_FALSE(se::descendant(ancestor_key, octant_key, voxel_depth));
}

TEST(Octree, OctantParent) {
  const int voxel_depth = 8;
  Eigen::Vector3i octant_coord = {112, 80, 160};
  se::key_t octant_key =
    se::keyops::encode(octant_coord.x(), octant_coord.y(), octant_coord.z(), 5, voxel_depth);
  se::key_t parent_key = se::parent(octant_key, voxel_depth);
  ASSERT_EQ(se::keyops::code(octant_key), se::keyops::code(parent_key));
  ASSERT_EQ(4u, parent_key & SCALE_MASK);

  octant_key = parent_key;
  parent_key = se::parent(octant_key, voxel_depth);
  ASSERT_EQ(3, se::keyops::depth(parent_key));
  ASSERT_EQ(parent_key, se::keyops::encode(96, 64, 160, 3, voxel_depth));

  octant_key = parent_key;
  parent_key = se::parent(octant_key, voxel_depth);
  ASSERT_EQ(2, se::keyops::depth(parent_key));
  ASSERT_EQ(parent_key, se::keyops::encode(64, 64, 128, 2, voxel_depth));
}

TEST(Octree, FarCorner) {
  /*
   * The far corner should always be "exterior", meaning that moving one step
   * in the outward direction (i.e. away from the centre) in *any* direction
   * implies leaving the parent octant. For simplicity here the corners
   * individually, but this can be done programmatically testing this property.
   * TODO: change this test case to be more exhaustive.
   */

  const int voxel_depth = 5;
  const int depth = 2;

  /* First child */
  const se::key_t octant_0_key =
    se::keyops::encode(16, 16, 16, depth, voxel_depth);
  const Eigen::Vector3i fc_0_coord = se::far_corner(octant_0_key, depth, voxel_depth);
  ASSERT_EQ(fc_0_coord.x(), 16);
  ASSERT_EQ(fc_0_coord.y(), 16);
  ASSERT_EQ(fc_0_coord.z(), 16);

  /* Second child */
  const se::key_t octant_1_key = se::keyops::encode(24, 16, 16, depth, voxel_depth);
  const Eigen::Vector3i fc_1_coord = se::far_corner(octant_1_key, depth, voxel_depth);
  ASSERT_EQ(fc_1_coord.x(), 32);
  ASSERT_EQ(fc_1_coord.y(), 16);
  ASSERT_EQ(fc_1_coord.z(), 16);

  /* Third child */
  const se::key_t octant_2_key = se::keyops::encode(16, 24, 16, depth, voxel_depth);
  const Eigen::Vector3i fc_2_coord = se::far_corner(octant_2_key, depth, voxel_depth);
  ASSERT_EQ(fc_2_coord.x(), 16);
  ASSERT_EQ(fc_2_coord.y(), 32);
  ASSERT_EQ(fc_2_coord.z(), 16);

  /* Fourth child */
  const se::key_t octant_3_key = se::keyops::encode(24, 24, 16, depth, voxel_depth);
  const Eigen::Vector3i fc_3_coord = se::far_corner(octant_3_key, depth, voxel_depth);
  ASSERT_EQ(fc_3_coord.x(), 32);
  ASSERT_EQ(fc_3_coord.y(), 32);
  ASSERT_EQ(fc_3_coord.z(), 16);

  /* Fifth child */
  const se::key_t octant_4_key = se::keyops::encode(24, 24, 16, depth, voxel_depth);
  const Eigen::Vector3i fc_4_coord = se::far_corner(octant_4_key, depth, voxel_depth);
  ASSERT_EQ(fc_4_coord.x(), 32);
  ASSERT_EQ(fc_4_coord.y(), 32);
  ASSERT_EQ(fc_4_coord.z(), 16);

  /* sixth child */
  const se::key_t octant_5_key = se::keyops::encode(16, 16, 24, depth, voxel_depth);
  const Eigen::Vector3i fc_5_coord = se::far_corner(octant_5_key, depth, voxel_depth);
  ASSERT_EQ(fc_5_coord.x(), 16);
  ASSERT_EQ(fc_5_coord.y(), 16);
  ASSERT_EQ(fc_5_coord.z(), 32);

  /* seventh child */
  const se::key_t octant_6_key = se::keyops::encode(24, 16, 24, depth, voxel_depth);
  const Eigen::Vector3i fc_6_coord = se::far_corner(octant_6_key, depth, voxel_depth);
  ASSERT_EQ(fc_6_coord.x(), 32);
  ASSERT_EQ(fc_6_coord.y(), 16);
  ASSERT_EQ(fc_6_coord.z(), 32);

  /* eight child */
  const se::key_t octant_7_key = se::keyops::encode(24, 24, 24, depth, voxel_depth);
  const Eigen::Vector3i fc_7_coord = se::far_corner(octant_7_key, depth, voxel_depth);
  ASSERT_EQ(fc_7_coord.x(), 32);
  ASSERT_EQ(fc_7_coord.y(), 32);
  ASSERT_EQ(fc_7_coord.z(), 32);
}

TEST(Octree, InnerOctantExteriorNeighbours) {
  const int voxel_depth = 5;
  const int depth = 2;
  const se::key_t octant_key = se::keyops::encode(16, 16, 16, depth, voxel_depth);
  se::key_t neighbour_keys[7];
  se::exterior_neighbours(neighbour_keys, octant_key, depth, voxel_depth);
  const se::key_t parent_key = se::parent(octant_key, voxel_depth);

  const se::key_t neighbours_gt_keys[7] =
    {se::keyops::encode(15, 16, 16, depth, voxel_depth),
     se::keyops::encode(16, 15, 16, depth, voxel_depth),
     se::keyops::encode(15, 15, 16, depth, voxel_depth),
     se::keyops::encode(16, 16, 15, depth, voxel_depth),
     se::keyops::encode(15, 16, 15, depth, voxel_depth),
     se::keyops::encode(16, 15, 15, depth, voxel_depth),
     se::keyops::encode(15, 15, 15, depth, voxel_depth)};
  for(int i = 0; i < 7; ++i) {
    ASSERT_EQ(neighbours_gt_keys[i], neighbour_keys[i]);
    ASSERT_FALSE(se::parent(neighbour_keys[i], voxel_depth) == parent_key);
  }
}

TEST(Octree, EdgeOctantExteriorNeighbours) {
  const int voxel_depth = 5;
  const int size = 1 << voxel_depth;
  const int depth = 2;
  const se::key_t octant_key = se::keyops::encode(0, 16, 16, depth, voxel_depth);
  se::key_t neightbour_keys[7];
  se::exterior_neighbours(neightbour_keys, octant_key, depth, voxel_depth);

  for(int i = 0; i < 7; ++i) {
    const Eigen::Vector3i neighbour_coord = unpack_morton(neightbour_keys[i] & ~SCALE_MASK);
    const int res = ((neighbour_coord.array() >= Eigen::Vector3i::Constant(0).array()) &&
                     (neighbour_coord.array() <= Eigen::Vector3i::Constant(size - 1).array())).all();
    ASSERT_TRUE(res);
  }
}

TEST(Octree, OctantSiblings) {
  const int voxel_depth = 5;
  const int depth = 2;
  const se::key_t octant_key = se::keyops::encode(16, 16, 16, depth, voxel_depth);
  se::key_t sibling_keys[8];
  se::siblings(sibling_keys, octant_key, voxel_depth);

  const int child_idx = se::child_idx(octant_key, depth, voxel_depth);
  ASSERT_EQ(sibling_keys[child_idx], octant_key);
  const Eigen::Vector3i octant_coord = se::keyops::decode(octant_key);
  for(int i = 0; i < 8; ++i) {
    // std::cout << (unpack_morton(s[i] & ~SCALE_MASK)) << std::endl;
    const Eigen::Vector3i parent_coord = se::keyops::decode(se::parent(sibling_keys[i], voxel_depth));
    ASSERT_TRUE(se::parent(sibling_keys[i], voxel_depth) == se::parent(octant_key, voxel_depth));
    ASSERT_TRUE(octant_coord.x() == parent_coord.x() &&
                octant_coord.y() == parent_coord.y() &&
                octant_coord.z() == parent_coord.z());
  }
}

TEST(Octree, OctantOneNeighbours) {
  const int voxel_depth = 8;
  const int depth = 5;
  const unsigned map_size = std::pow(2, voxel_depth);
  Eigen::Matrix<int, 4, 6> N;
  Eigen::Vector3i voxel_coord;

  // Inside cube
  //
  voxel_coord << 127, 56, 3;
  se::one_neighbourhood(N, se::keyops::encode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(),
        depth, voxel_depth), voxel_depth);
  ASSERT_TRUE((N.array() >= 0).all() && (N.array() < map_size).all());


  // At edge cube
  //
  voxel_coord << map_size-1, 56, 3;
  se::one_neighbourhood(N, se::keyops::encode(voxel_coord.x(), voxel_coord.y(), voxel_coord.z(),
        depth, voxel_depth), voxel_depth);
  ASSERT_TRUE((N.array() >= 0).all() && (N.array() < map_size).all());
}

TEST(Octree, BalanceTree) {
  const int voxel_depth = 8;

  se::Octree<TestVoxelT> octree;
  octree.init(1 << voxel_depth, 5);
  Eigen::Vector3i voxel_coord(32, 208, 44);
  octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
  voxel_coord += Eigen::Vector3i(50, 12, 100);
  octree.insert(voxel_coord.x(), voxel_coord.y(), voxel_coord.z());
  se::save_octree_structure_ply(octree, "./oct.ply");
  se::balance(octree);
  se::save_octree_structure_ply(octree, "./oct-balanced.ply");
}
