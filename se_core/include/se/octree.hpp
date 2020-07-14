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

#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <cstring>
#include <algorithm>
#include <vector>
#include "utils/math_utils.h"
#include "octree_defines.h"
#include "utils/morton_utils.hpp"
#include "octant_ops.hpp"

#if defined(_OPENMP) && !defined(__clang__)
#include <parallel/algorithm>
#endif

#include <array>
#include <tuple>
#include <queue>
#include <unordered_set>
#include "node.hpp"
#include "utils/memory_pool.hpp"
#include "algorithms/unique.hpp"
#include "geometry/aabb_collision.hpp"
#include "interpolation/interp_gather.hpp"
#include "se/interpolation/interp_gather.hpp"
#include "neighbors/neighbor_gather.hpp"

namespace se {

/*
 * Value between 0.f and 1.f. Defines the sample point position relative to the
 * voxel anchor.  E.g. 0.5f means that the point sample corresponds to the
 * centre of the voxel.
 */
#define SAMPLE_POINT_POSITION 0.5f

template <typename T>
class VoxelBlockRayIterator;

template <typename T>
class node_iterator;

/*! \brief The main octree class.
 * Its non-leaf nodes are of type Node and its leaf nodes of type VoxelBlock.
 * For a minimal working example of the kind of struct needed as a template
 * parameter see ExampleVoxelT.
 */
template <typename T>
class Octree {
  typedef typename T::VoxelData VoxelData;
  using VoxelBlockType = typename T::VoxelBlockType;

public:
  // Compile-time constant expressions
  // # of voxels per side in a voxel block
  static constexpr unsigned int block_size = BLOCK_SIZE;
  // maximum tree depth in bits
  static constexpr unsigned int max_voxel_depth = ((sizeof(key_t) * 8) / 3);
  // Tree depth at which blocks are found
  static constexpr unsigned int max_block_depth = max_voxel_depth - math::log2_const(BLOCK_SIZE);

  static const Eigen::Vector3f sample_offset_frac_;


  Octree(){
  };

  ~Octree(){
  }

  /*! \brief Initialises the octree attributes
   * \param size number of voxels per side of the cube
   * \param dim cube extension per side, in meter
   */
  void init(int size, float dim);

  inline int size() const { return size_; }
  inline float dim() const { return dim_; }
  inline float voxelDim() const { return voxel_dim_; }
  inline float inverseVoxelDim() const { return inverse_voxel_dim_; }
  inline int numLevels() const { return num_levels_; }
  inline int voxelDepth() const { return voxel_depth_; }
  inline int maxBlockScale() const { return max_block_scale_; }
  inline int blockDepth() const { return block_depth_; }
  inline Node<T>* root() const { return root_; }

  /*! \brief Verify if the each coordinate x, y and z is in the interval [0, size - 1]
   *
   * \param[in] x The voxel x coordinate
   * \param[in] y The voxel y coordinate
   * \param[in] z The voxel z coordinate
   * \return    True if each coordinate x, y and z is in the interval [0, size - 1]; False otherwise.
   */
  inline bool contains(const int x, const int y, const int z) const;

  /*! \brief Verify if each voxel coordinate is in the interval [0, size - 1]
   *
   * \param[in] voxel_coord The coordinates of the voxel.
   * \return    True if each voxel coordinate is in the interval [0, size - 1]; False otherwise.
   */
  inline bool contains(Eigen::Vector3i& voxel_coord) const;

  /*! \brief Verify if each voxel coordinate is in the interval [0, size)
   *
   * \param[in] voxel_coord_f The coordinates of the voxel.
   * \return    True if each voxel coordinate is in the interval [0, size); False otherwise.
   */
  inline bool contains(Eigen::Vector3f& voxel_coord_f) const;

  /*! \brief Verify if each point coordinate is in the interval [0, dim)
   *
   * \param[in] point_M The coordinates of the point.
   * \return    True if each point coordinate is in the interval [0, dim); False otherwise.
   */
  inline bool containsPoint(Eigen::Vector3f& point_M) const;

  /*! \brief Set the data at the supplied voxel coordinates.
   * If the voxel hasn't been allocated, no action is performed.
   *
   * \param[in] x    The voxel x coordinate in the interval [0, size - 1].
   * \param[in] y    The voxel y coordinate in the interval [0, size - 1].
   * \param[in] z    The voxel z coordinate in the interval [0, size - 1].
   * \param[in] data The data to store in the voxel.
   */
  void set(const int x, const int y, const int z, const VoxelData& data);

  /*! \brief Set the data at the supplied voxel coordinates.
   * If the voxel hasn't been allocated, no action is performed.
   *
   * \param[in] voxel_coord The coordinates of the voxel. Each component must
   *                        be in the interval [0, size - 1].
   * \param[in] data The data to store in the voxel.
   */
  void set(const Eigen::Vector3i& voxel_coord, const VoxelData& data);

  /*! \brief Set the data at the supplied 3D point.
   * If the voxel hasn't been allocated, no action is performed.
   *
   * \param[in] point_M The coordinates of the point. Each component must be in
   *                    the interval [0, dim).
   * \param[in] data The data to store in the voxel.
   */
  void setAtPoint(const Eigen::Vector3f& point_M, const VoxelData& data);

  /*! \brief Return the data at the supplied voxel coordinates.
   *
   * \param[in] x The voxel x coordinate in the interval [0, size - 1].
   * \param[in] y The voxel y coordinate in the interval [0, size - 1].
   * \param[in] z The voxel z coordinate in the interval [0, size - 1].
   * \return The data contained in the voxel. If the octree hasn't been
   *         allocated up to the voxel level at this region return the value
   *         stored at the lowest allocated octant.
   */
  VoxelData get(const int x, const int y, const int z) const;

  /*! \brief Return the data at the supplied voxel coordinates.
   *
   * \param[in] voxel_coord The coordinates of the voxel. Each component must
   *                        be in the interval [0, size - 1].
   * \return The data contained in the voxel. If the octree hasn't been
   *         allocated up to the voxel level at this region return the value
   *         stored at the lowest allocated octant.
   */
  VoxelData get(const Eigen::Vector3i& voxel_coord) const;

  /*! \brief Return the data at the supplied 3D point.
   *
   * \param[in] point_M The coordinates of the point. Each component must be in
   *                    the interval [0, dim).
   * \return The data contained in the corresponding voxel. If the octree
   *         hasn't been allocated up to the voxel level at this region return
   *         the value stored at the lowest allocated octant.
   */
  VoxelData getAtPoint(const Eigen::Vector3f& point_M) const;

  /*! \brief Return the data at the supplied voxel coordinates and scale.
   *
   * \param[in] x     The voxel x coordinate in the interval [0, size - 1].
   * \param[in] y     The voxel y coordinate in the interval [0, size - 1].
   * \param[in] z     The voxel z coordinate in the interval [0, size - 1].
   * \param[in] scale The octree scale to get the data at.
   * \return The data contained in the voxel. If the octree hasn't been
   *         allocated up to the supplied scale, return the data at the lowest
   *         allocated scale.
   */
  VoxelData getFine(const int x, const int y, const int z, const int scale = 0) const;

  /*! \brief Return the data at the supplied voxel coordinates and scale.
   *
   * \param[in] voxel_coord The coordinates of the voxel. Each component must
   *                        be in the interval [0, size - 1].
   * \param[in] scale The octree scale to get the data at.
   * \return The data contained in the voxel. If the octree hasn't been
   *         allocated up to the supplied scale, return the data at the lowest
   *         allocated scale.
   */
  VoxelData getFine(const Eigen::Vector3i& voxel_coord, const int scale = 0) const;

  /*! \brief Return the data at the supplied 3D point and scale.
   *
   * \param[in] point_M The coordinates of the point. Each component must be in
   *                    the interval [0, dim).
   * \param[in] scale   The octree scale to get the data at.
   * \return The data contained in corresponding the voxel. If the octree
   *         hasn't been allocated up to the supplied scale, return the data
   *         at the lowest allocated scale.
   */
  VoxelData getFineAtPoint(const Eigen::Vector3f& point_M, const int scale = 0) const;

  /*! \brief Convert voxel coordinates to the coordinates of the correspoinding
   * 3D point in metres.
   *
   * \param voxel_coord The voxel coordinates.
   * \return The coordinates of the corresponding 3D point in metres.
   */
  inline Eigen::Vector3f voxelToPoint(const Eigen::Vector3i& voxel_coord) const;

  /*! \brief Convert voxel coordinates to the coordinates of the correspoinding
   * 3D point in metres.
   *
   * \param voxel_coord_f The voxel coordinates.
   * \return The coordinates of the corresponding 3D point in metres.
   */
  inline Eigen::Vector3f voxelToPoint(const Eigen::Vector3f& voxel_coord_f) const;

  /*! \brief Convert 3D point coordinates in metres to the coordinates of the
   * corresponding voxel.
   *
   * \param point_M The coordinates of the 3D point in metres. Each component must
   *                be in the interval [0, dim).
   * \return The corresponding int voxel coordinates.
   */
  inline Eigen::Vector3i pointToVoxel(const Eigen::Vector3f& point_M) const;

  /*! \brief Convert 3D point coordinates in metres to the coordinates of the
   * corresponding voxel.
   *
   * \param point_M The coordinates of the 3D point in metres. Each component must
   *                be in the interval [0, dim).
   * \return The corresponding float voxel coordinates.
   */
  inline Eigen::Vector3f pointToVoxelF(const Eigen::Vector3f& point_M) const;

  /*! \brief Retrieves voxel values for the neighbors of voxel at coordinates
   * (x,y,z)
   * If the safe template variable is true, then proper checks will be used so
   * that neighboring voxels outside the map will have a value of empty at a
   * cost of performance. Otherwise if the safe template variable is false,
   * neighboring voxels outside the map will not be detected and will have the
   * value of some voxel inside the map. If you are certain your code will not
   * ask for the neighbors of voxels at the edge of the map, then the non-safe
   * version should be safe to use and will result in better performance.
   *
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   * \return An std::array with the values of the 6 neighboring voxels. The
   * voxels are returned in the order: -z -y -x +x +y +z. Neighboring voxels
   * that are not allocated have the initial value. Neighboring voxels that are
   * outside the map have the empty value if safe is true, otherwise their
   * value is undetermined.
   *
   * \todo The implementation is not yet efficient. A method similar to the one
   * used in interp_gather should be used.
   */
  template <bool safe>
  std::array<VoxelData, 6> get_face_neighbors(const int x,
                                              const int y,
                                              const int z) const;

  /*! \brief Fetch the voxel block which contains voxel (x,y,z)
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   */
  VoxelBlockType* fetch(const int x, const int y, const int z) const;

  /*! \brief Fetch the voxel block which contains voxel (x,y,z)
   * \param[in] voxel_coord The coordinates of the voxel. Each component must
   *                        be in the interval [0, size).
   */
  VoxelBlockType* fetch(const Eigen::Vector3i& voxel_coord) const;

  /*! \brief Fetch the node (x,y,z) at depth
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   * \param depth depth to be searched
   */
  Node<T>* fetch_node(const int x, const int y, const int z,
      const int depth) const;

  /*! \brief Fetch the node (x,y,z) at depth
   * \param[in] voxel_coord The coordinates of the voxel. Each component must
   *                        be in the interval [0, size).
   * \param[in] depth depth to be searched
   */
  Node<T>* fetch_node(const Eigen::Vector3i& voxel_coord,
                      const int depth) const;

  /*! \brief Insert the octant at (x,y,z). Not thread safe.
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   * \param depth target insertion depth
   * \param init_octant optional inital state of inserted node / voxel block
   */
  Node<T>* insert(const int x,
                  const int y,
                  const int z,
                  const int depth,
                  Node<T>*  init_octant = nullptr);

  /*! \brief Insert the block (x,y,z) at maximum resolution. Not thread safe.
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   * \param init_block optional inital state of inserted voxel block
   */
  VoxelBlockType* insert(const int        x,
                         const int        y,
                         const int        z,
                         VoxelBlockType*  init_block = nullptr);

  /*! \brief Interpolate a voxel value at the supplied voxel coordinates.
   *
   * \param[in] voxel_coord_f The coordinates of the voxel. Each component must
   *                          be in the interval [0, size).
   * \param[in] select_value  Lambda value to select the value to compute the
   *                          gradient for from the voxel data.
   * \param[in] min_scale     The minimum scale at which the interpolation is
   *                          performed. Must be at least 1 because the
   *                          interpolation can't be performed with a single
   *                          point.
   * \return The interpolated value.
   */
  template <typename ValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               ValueSelector          select_value,
                               const int              min_scale = 0) const;

  /*! \brief Interpolate a voxel value at the supplied voxel coordinates.
   *
   * \param[in]  voxel_coord_f The coordinates of the voxel. Each component
   *                           must be in the interval [0, size).
   * \param[in]  select_value  Lambda value to select the value to compute the
   *                           gradient for from the voxel data.
   * \param[in]  min_scale     The minimum scale at which the interpolation is
   *                           performed. Must be at least 1 because the
   *                           interpolation can't be performed with a single
   *                           point.
   * \param[out] is_valid      False when the interpolation uses data from
   *                           voxels which haven't been integrated into.
   * \return The interpolated value.
   */
  template <typename ValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               ValueSelector          select_value,
                               const int              min_scale,
                               bool&                  is_valid) const;

  /*! \brief Interpolate a voxel value at the supplied voxel coordinates.
   *
   * \param[in] voxel_coord_f      The coordinates of the voxel. Each component
   *                               must be in the interval [0, size).
   * \param[in] select_node_value  Lambda value to select the value to compute
   *                               the gradient for from the voxel data.
   * \param[in] select_voxel_value Lambda value to select the value to compute
   *                               the gradient for from the node data.
   * \param[in] min_scale          The minimum scale at which the interpolation
   *                               is performed. Must be at least 1 because the
   *                               interpolation can't be performed with a
   *                               single point.
   * \return The interpolated value.
   */
  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               NodeValueSelector      select_node_value,
                               VoxelValueSelector     select_voxel_value,
                               const int              min_scale = 0) const;

  /*! \brief Interpolate a voxel value at the supplied voxel coordinates.
   *
   * \param[in]  voxel_coord_f      The coordinates of the voxel. Each
   *                                component must be in the interval [0,
   *                                size).
   * \param[in]  select_node_value  Lambda value to select the value to compute
   *                                the gradient for from the voxel data.
   * \param[in]  select_voxel_value Lambda value to select the value to compute
   *                                the gradient for from the node data.
   * \param[in]  min_scale          The minimum scale at which the
   *                                interpolation is performed. Must be at
   *                                least 1 because the interpolation can't be
   *                                performed with a single point.
   * \param[out] is_valid           False when the interpolation uses data from
   *                                voxels which haven't been integrated into.
   * \return The interpolated value.
   */
  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interp(const Eigen::Vector3f& voxel_coord_f,
                               NodeValueSelector      select_node_value,
                               VoxelValueSelector     select_voxel_value,
                               const int              min_scale,
                               bool&                  is_valid) const;

  /*! \brief Interpolate a voxel value at the supplied 3D point.
   *
   * \param[in] point_M      The coordinates of the point. Each component must
   *                         be in the interval [0, dim).
   * \param[in] select_value Lambda value to select the value to compute the
   *                         gradient for from the voxel data.
   * \param[in] min_scale    The minimum scale at which the interpolation is
   *                         performed. Must be at least 1 because the
   *                         interpolation can't be performed with a single
   *                         point.
   * \return The interpolated value.
   */
  template <typename ValueSelector>
  std::pair<float, int> interpAtPoint(const Eigen::Vector3f& point_M,
                                      ValueSelector          select_value,
                                      const int              min_scale = 0) const;

  /*! \brief Interpolate a voxel value at the supplied 3D point.
   *
   * \param[in]  point_M      The coordinates of the point. Each component
   *                          must be in the interval [0, dim).
   * \param[in]  select_value Lambda value to select the value to compute the
   *                          gradient for from the voxel data.
   * \param[in]  min_scale    \TODO Document this
   * \param[out] is_valid     False when the interpolation uses data from
   *                          voxels which haven't been integrated into.
   * \return The interpolated value.
   */
  template <typename ValueSelector>
  std::pair<float, int> interpAtPoint(const Eigen::Vector3f& point_M,
                                      ValueSelector          select_value,
                                      const int              min_scale,
                                      bool&                  is_valid) const;

  /*! \brief Interpolate a voxel value at the supplied 3D point.
   *
   * \param[in] point_M            The coordinates of the point. Each component
   *                               must be in the interval [0, dim).
   * \param[in] select_node_value  Lambda value to select the value to compute
   *                               the gradient for from the voxel data.
   * \param[in] select_voxel_value Lambda value to select the value to compute
   *                               the gradient for from the node data.
   * \param[in] min_scale          \TODO Document this
   * \return The interpolated value.
   */
  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interpAtPoint(const Eigen::Vector3f& point_M,
                                      NodeValueSelector      select_node_value,
                                      VoxelValueSelector     select_voxel_value,
                                      const int              min_scale = 0) const;

  /*! \brief Interpolate a voxel value at the supplied 3D point.
   *
   * \param[in]  point_M            The coordinates of the point. Each
   *                                component must be in the interval [0, dim).
   * \param[in]  select_node_value  Lambda value to select the value to compute
   *                                the gradient for from the voxel data.
   * \param[in]  select_voxel_value Lambda value to select the value to compute
   *                                the gradient for from the node data.
   * \param[in]  min_scale          \TODO Document this
   * \param[out] is_valid           False when the interpolation uses data from
   *                                voxels which haven't been integrated into.
   * \return The interpolated value.
   */
  template <typename NodeValueSelector, typename VoxelValueSelector>
  std::pair<float, int> interpAtPoint(const Eigen::Vector3f& point_M,
                                      NodeValueSelector      select_node_value,
                                      VoxelValueSelector     select_voxel_value,
                                      const int              min_scale,
                                      bool&                  is_valid) const;


  /*! \brief Compute the gradient of a voxel value at the supplied voxel
   * coordinates.
   *
   * \param[in] voxel_coord_f The coordinates of the voxel. Each component must
   *                          be in the interval [0, size).
   * \param[in] select_value  Lambda value to select the value to compute the
   *                          gradient for from the voxel data.
   * \param[in] min_scale     The minimum scale at which the gradient is
   *                          computed. Must be at least 1 because the gradient
   *                          can't be computed from a single point.
   * \return The gradient of the selected value.
   */
  template <typename FieldSelect>
  Eigen::Vector3f grad(const Eigen::Vector3f& voxel_coord_f,
                       FieldSelect            select_value,
                       const int              min_scale = 1) const;

  /*! \brief Compute the gradient of a voxel value at the supplied 3D point.
   *
   * \param[in] point_M       The coordinates of the 3D point in metres.
   * \param[in] select_value  Lambda value to select the value to compute the
   *                          gradient for from the voxel data.
   * \param[in] min_scale     The minimum scale at which the gradient is
   *                          computed. Must be at least 1 because the gradient
   *                          can't be computed from a single point.
   * \return The gradient of the selected value.
   */
  template <typename FieldSelect>
  Eigen::Vector3f gradAtPoint(const Eigen::Vector3f& point_M,
                              FieldSelect            select_value,
                              const int              min_scale = 1) const;

  /*! \brief Get the list of allocated block. If the active switch is set to
   * true then only the visible blocks are retrieved.
   * \param block_list output vector of allocated blocks
   * \param active boolean switch. Set to true to retrieve visible, allocated
   * blocks, false to retrieve all allocated blocks.
   */
  void getBlockList(std::vector<VoxelBlockType *>& block_list, bool active);
  typename T::MemoryPoolType& pool() { return pool_; };
  const typename T::MemoryPoolType& pool() const { return pool_; };

  /*! \brief Computes the morton code of the block containing voxel
   * at coordinates (x,y,z)
   * \param x x coordinate in interval [0, size - 1]
   * \param y y coordinate in interval [0, size - 1]
   * \param z z coordinate in interval [0, size - 1]
   */
  key_t hash(const int x, const int y, const int z) {
    const int scale = voxel_depth_ - math::log2_const(block_size);
    return keyops::encode(x, y, z, scale, voxel_depth_);
  }

  key_t hash(const int x, const int y, const int z, key_t scale) {
    return keyops::encode(x, y, z, scale, voxel_depth_);
  }

  /*! \brief allocate a set of voxel blocks via their positional key
   * \param keys collection of voxel block keys to be allocated (i.e. their
   * morton number)
   * \param number of keys in the keys array
   */
  bool allocate(key_t *keys, int num_elem);

  void save(const std::string& filename);
  void load(const std::string& filename);

  /*! \brief Counts the number of blocks allocated
   * \return number of voxel blocks allocated
   */
  int blockCount();

  /*! \brief Counts the number of internal nodes
   * \return number of internal nodes
   */
  int nodeCount();

  void printMemStats(){
    // memory.printStats();
  };

private:

  Node<T>* root_ = nullptr;
  int size_ = 0;
  float dim_ = 0.f;
  float voxel_dim_ = 0.0f;
  float inverse_voxel_dim_ = 0.0f;
  int num_levels_ = 0;
  int voxel_depth_ = 0;
  int max_block_scale_ = 0;
  int block_depth_ = 0;
  typename T::MemoryPoolType pool_;

  friend class VoxelBlockRayIterator<T>;
  friend class node_iterator<T>;

  // Allocation specific variables
  std::vector<key_t> keys_at_depth_;
  int reserved_ = 0;

  // Private implementation of cached methods
  VoxelData get(const int x, const int y, const int z, VoxelBlockType* cached) const;
  VoxelData getAtPoint(const Eigen::Vector3f& point_M, VoxelBlockType* cached) const;

  VoxelData get(const int x, const int y, const int z,
      int&  scale, VoxelBlockType* cached) const;
  VoxelData getAtPoint(const Eigen::Vector3f& point_M, int& scale,
      VoxelBlockType* cached) const;

  // Parallel allocation of a given tree depth for a set of input keys.
  // Pre: depth above target_depth must have been already allocated
  bool allocate_depth(key_t * keys, int num_tasks, int target_depth);

  void reserveBuffers(const int n);

  // General helpers

  int blockCountRecursive(Node<T>*);
  int nodeCountRecursive(Node<T>*);
  void getActiveBlockList(Node<T>*, std::vector<VoxelBlockType *>& block_list);
  void getAllocatedBlockList(Node<T>*, std::vector<VoxelBlockType *>& block_list);

  void deleteNode(Node<T>** node);
  void deallocateTree(){ deleteNode(&root_); }
};

}

#include "octree_impl.hpp"

#endif // OCTREE_HPP
