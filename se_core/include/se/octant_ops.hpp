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
#ifndef OCTANT_OPS_HPP
#define OCTANT_OPS_HPP
#include "utils/morton_utils.hpp"
#include "utils/math_utils.h"
#include "octree_defines.h"
#include <iostream>
#include <bitset>
#include <Eigen/Dense>

namespace se {
  namespace keyops {

    inline se::key_t code(const se::key_t octant_key) {
      return octant_key & ~SCALE_MASK;
    }

    inline int depth(const se::key_t octant_key) {
      return octant_key & SCALE_MASK;
}

    inline se::key_t encode(const int x, const int y, const int z,
        const int depth, const int voxel_depth) {
      const int offset = MAX_BITS - voxel_depth + depth - 1;
      return (compute_morton(x, y, z) & MASK[offset] & ~SCALE_MASK) | depth;
    }

    inline Eigen::Vector3i decode(const se::key_t octant_key) {
      return unpack_morton(octant_key & ~SCALE_MASK);
    }
  }

/*
 * Algorithm 5 of p4est paper: https://epubs.siam.org/doi/abs/10.1137/100791634
 */
inline Eigen::Vector3i face_neighbour(const se::key_t octant_key,
    const unsigned int face, const unsigned int l,
    const unsigned int voxel_depth) {
  Eigen::Vector3i octant_coord = se::keyops::decode(octant_key);
  const unsigned int octant_size = 1 << (voxel_depth - l);
  octant_coord.x() = octant_coord.x() + ((face == 0) ? -octant_size : (face == 1) ? octant_size : 0);
  octant_coord.y() = octant_coord.y() + ((face == 2) ? -octant_size : (face == 3) ? octant_size : 0);
  octant_coord.z() = octant_coord.z() + ((face == 4) ? -octant_size : (face == 5) ? octant_size : 0);
  return {octant_coord.x(), octant_coord.y(), octant_coord.z()};
}

/*
 * \brief Return true if node is a descendant of ancestor
 * \param[in] octant_key
 * \param[in] ancestor_key
 * \param[in] voxel_depth  Max/voxel depth of the tree the keys refer to.
 * \return True if octant_key is a descendant of the ancestor_key or if
 *         they are equal.
 */
inline bool descendant(const se::key_t octant_key,
                       const se::key_t ancestor_key,
                       const int       voxel_depth) {
  const int depth = se::keyops::depth(octant_key);
  const int ancestor_depth = se::keyops::depth(ancestor_key);
  // MAX_BITS is used for the maximum voxel depth given the size of the se::key_t
  const int idx = MAX_BITS - voxel_depth + ancestor_depth - 1;
  const se::key_t ancestor_key_code = se::keyops::code(ancestor_key);
  const se::key_t octant_key_code = se::keyops::code(octant_key) & MASK[idx];
  return (depth >= ancestor_depth) && (ancestor_key_code ^ octant_key_code) == 0;
}

/*
 * \brief Computes the parent's morton code of a given octant
 * \param octant_key
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline se::key_t parent(const se::key_t& octant_key, const int voxel_depth) {
  const int depth = se::keyops::depth(octant_key) - 1;
  const int idx = MAX_BITS - voxel_depth + depth - 1;
  return (octant_key & MASK[idx]) | depth;
}

/*
 * \brief Computes the octants's id in its local brotherhood
 * \param octant_key
 * \param depth of octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline int child_idx(se::key_t octant_key, const int depth,
    const int voxel_depth) {
  int shift = voxel_depth - depth;
  octant_key = se::keyops::code(octant_key) >> shift*3;
  int idx = (octant_key & 0x01) | (octant_key & 0x02) | (octant_key & 0x04);
  return idx;
}

/*
 * \brief Computes the octants's id in its local brotherhood
 * \param octant_key
 * \param depth of octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline int child_idx(se::key_t octant_key, const int voxel_depth) {
  int shift = voxel_depth - se::keyops::depth(octant_key);
  octant_key = se::keyops::code(octant_key) >> shift*3;
  int idx = (octant_key & 0x01) | (octant_key & 0x02) | (octant_key & 0x04);
  return idx;
}

/*
 * \brief Computes the octants's corner which is not shared with its siblings
 * \param octant_key
 * \param depth of octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline Eigen::Vector3i far_corner(const se::key_t octant_key, const int depth,
    const int voxel_depth) {
  const unsigned int octant_size = 1 << (voxel_depth - depth);
  const int child_idx = se::child_idx(octant_key, depth, voxel_depth);
  const Eigen::Vector3i octant_coord = se::keyops::decode(octant_key);
  return Eigen::Vector3i(octant_coord.x() + ( child_idx & 1)       * octant_size,
                         octant_coord.y() + ((child_idx & 2) >> 1) * octant_size,
                         octant_coord.z() + ((child_idx & 4) >> 2) * octant_size);
}

/*
 * \brief Computes the non-sibling neighbourhood around an octants. In the
 * special case in which the octant lies on an edge, neighbour are duplicated
 * as movement outside the enclosing cube is forbidden.
 * \param result 7-vector containing the neighbours
 * \param octant_key
 * \param depth of octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline void exterior_neighbours(se::key_t result[7],
    const se::key_t octant_key, const int depth, const int voxel_depth) {

  const int child_idx = se::child_idx(octant_key, depth, voxel_depth);
  Eigen::Vector3i dir = Eigen::Vector3i((child_idx & 1) ? 1 : -1,
                                        (child_idx & 2) ? 1 : -1,
                                        (child_idx & 4) ? 1 : -1);
  Eigen::Vector3i base_coord = far_corner(octant_key, depth, voxel_depth);
  dir.x() = se::math::in(base_coord.x() + dir.x() , 0, (1 << voxel_depth) - 1) ? dir.x() : 0;
  dir.y() = se::math::in(base_coord.y() + dir.y() , 0, (1 << voxel_depth) - 1) ? dir.y() : 0;
  dir.z() = se::math::in(base_coord.z() + dir.z() , 0, (1 << voxel_depth) - 1) ? dir.z() : 0;

 result[0] = se::keyops::encode(base_coord.x() + dir.x(), base_coord.y() + 0, base_coord.z() + 0,
     depth, voxel_depth);
 result[1] = se::keyops::encode(base_coord.x() + 0, base_coord.y() + dir.y(), base_coord.z() + 0,
     depth, voxel_depth);
 result[2] = se::keyops::encode(base_coord.x() + dir.x(), base_coord.y() + dir.y(), base_coord.z() + 0,
     depth, voxel_depth);
 result[3] = se::keyops::encode(base_coord.x() + 0, base_coord.y() + 0, base_coord.z() + dir.z(),
     depth, voxel_depth);
 result[4] = se::keyops::encode(base_coord.x() + dir.x(), base_coord.y() + 0, base_coord.z() + dir.z(),
     depth, voxel_depth);
 result[5] = se::keyops::encode(base_coord.x() + 0, base_coord.y() + dir.y(), base_coord.z() + dir.z(),
     depth, voxel_depth);
 result[6] = se::keyops::encode(base_coord.x() + dir.x(), base_coord.y() + dir.y(),
     base_coord.z() + dir.z(), depth, voxel_depth);
}

/*
 * \brief Computes the six face neighbours of an octant. These are stored in an
 * 4x6 matrix in which each column represents the homogeneous coordinates of a
 * neighbouring octant. The neighbours along the x axis come first, followed by
 * neighbours along the y axis and finally along the z axis. All coordinates are
 * clamped to be in the range between [0, max_size] where max size is given
 * by pow(2, voxel_depth).
 * \param res 4x6 matrix containing the neighbours
 * \param octant_coord octant coordinates
 * \param depth depth of the octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */

static inline void one_neighbourhood(Eigen::Ref<Eigen::Matrix<int, 4, 6>> res,
    const Eigen::Vector3i& octant_coord, const int depth, const int voxel_depth) {
  const Eigen::Vector3i base_coord = octant_coord;
  const int size = 1 << voxel_depth;
  const int step = 1 << (voxel_depth - depth);
  Eigen::Matrix<int, 4, 6> cross;
  res <<
    -step, step,     0,    0,     0,    0,
        0,    0, -step, step,     0,    0,
        0,    0,     0,    0, -step, step,
        0,    0,     0,    0,     0,    0;
    res.colwise() += base_coord.homogeneous();
    res = res.unaryExpr([size](const int a) {
        return std::max(std::min(a, size-1), 0);
        });
}

/*
 * \brief Computes the six face neighbours of an octant. These are stored in an
 * 4x6 matrix in which each column represents the homogeneous coordinates of a
 * neighbouring octant. The neighbours along the x axis come first, followed by
 * neighbours along the y axis and finally along the z axis. All coordinates are
 * clamped to be in the range between [0, max_size] where max size is given
 * by pow(2, voxel_depth).
 * \param res 4x6 matrix containing the neighbours
 * \param octant octant key
 * \param voxel_depth max depth of the tree on which the octant lives
 */

static inline void one_neighbourhood(Eigen::Ref<Eigen::Matrix<int, 4, 6>> res,
    const se::key_t octant_key, const int voxel_depth) {
  one_neighbourhood(res, se::keyops::decode(octant_key), se::keyops::depth(octant_key),
      voxel_depth);
}

/*
 * \brief Computes the morton number of all siblings around an octant,
 * including itself.
 * \param result 8-vector containing the neighbours
 * \param octant
 * \param voxel_depth max depth of the tree on which the octant lives
 */
inline void siblings(se::key_t result[8],
    const se::key_t octant_key, const int voxel_depth) {
  const int depth = (octant_key & SCALE_MASK);
  const int shift = 3 * (voxel_depth - depth);
  const se::key_t parent_key = parent(octant_key, voxel_depth) + 1; // set-up next depth
  for(int i = 0; i < 8; ++i) {
    result[i] = parent_key | (i << shift);
  }
}
}
#endif
