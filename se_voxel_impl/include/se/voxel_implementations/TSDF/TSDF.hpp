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
#ifndef __TSDF_HPP
#define __TSDF_HPP

#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/sensor_implementation.hpp"

#include <yaml-cpp/yaml.h>

/**
 * Kinect Fusion Truncated Signed Distance Function voxel implementation.
 */
struct TSDF {

  /**
   * The voxel type used as the template parameter for se::Octree.
   */
  struct VoxelType {
    /**
     * The struct stored in each se::Octree voxel.
     */
    struct VoxelData {
      float x; /**< The value of the TSDF. */
      float y; /**< The number of measurements integrated in the voxel. */

      bool operator==(const VoxelData& other) const;
      bool operator!=(const VoxelData& other) const;
    };

    static inline VoxelData invalid()     { return {1.f, -1.f}; }
    static inline VoxelData initData() { return {1.f,  0.f}; }

    static float selectNodeValue(const VoxelData& /* data */) {
      return VoxelType::initData().x;
    }

    static float selectVoxelValue(const VoxelData& data) {
      return data.x;
    }

    static bool isInside(const VoxelData& data) {
      return data.x < 0.f;
    }

    static bool isValid(const VoxelData& data) {
      return (data.y > 0);
    }

    using VoxelBlockType = se::VoxelBlockFinest<TSDF::VoxelType>;

    using MemoryPoolType = se::PagedMemoryPool<TSDF::VoxelType>;
    template <typename ElemT>
    using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
  };

  using VoxelData      = TSDF::VoxelType::VoxelData;
  using OctreeType     = se::Octree<TSDF::VoxelType>;
  using VoxelBlockType = typename TSDF::VoxelType::VoxelBlockType;

  /**
   * The normals must be inverted when rendering a TSDF map.
   */
  static constexpr bool invert_normals = true;

  /**
   * The factor the voxel dim is multiplied with to compute mu
   *
   *  <br>\em Default: 8
   */
  static float mu_factor;

  /**
   * The TSDF truncation bound. Values of the TSDF are assumed to be in the
   * interval ±mu. See Section 3.3 of \cite NewcombeISMAR2011 for more
   * details.
   *  <br>\em Default: 8 x voxel_dim
   */
  static float mu;

  /**
   * The maximum value of the weight factor TSDF::VoxelType::VoxelData::y.
   */
  static float max_weight;

  static std::string type() { return "tsdf"; }

  /**
   * Configure the TSDF parameters
   */
  static void configure(const float voxel_dim);
  static void configure(YAML::Node yaml_config, const float voxel_dim);

  static std::string printConfig();

  /**
   * Compute the VoxelBlocks and Nodes that need to be allocated given the
   * camera pose.
   */
  static size_t buildAllocationList(OctreeType&             map,
                                    const se::Image<float>& depth_image,
                                    const Eigen::Matrix4f&  T_MC,
                                    const SensorImpl&       sensor,
                                    se::key_t*              allocation_list,
                                    size_t                  reserved);

  static size_t buildAllocationListFromRangeMeasurements(
      OctreeType& map, const std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>>& ranges,
      const Eigen::Matrix4f&  T_MC, se::key_t* allocation_list, size_t reserved) {return 0;}

  /**
   * Integrate a depth image into the map.
   */
  static void integrate(OctreeType&             map,
                        const se::Image<float>& depth_image,
                        const Eigen::Matrix4f&  T_CM,
                        const SensorImpl&       sensor,
                        const unsigned          frame, int weight = 1);

  static void integrateRangeMeasurements(
      OctreeType& map, const std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>>& ranges,
      const Eigen::Matrix4f& T_CM, const unsigned frame) {

  }

  /**
   * Cast a ray and return the point where the surface was hit.
   */
  static Eigen::Vector4f raycast(const OctreeType&      map,
                                 const Eigen::Vector3f& ray_origin_M,
                                 const Eigen::Vector3f& ray_dir_M,
                                 const float            t_near,
                                 const float            t_far);

  static void dumpMesh(OctreeType&                map,
                       std::vector<se::Triangle>& mesh);

};

#endif

