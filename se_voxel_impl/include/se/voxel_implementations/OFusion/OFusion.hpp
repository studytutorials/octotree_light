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
#ifndef __OFUSION_HPP
#define __OFUSION_HPP

#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/algorithms/meshing.hpp"
#include "se/sensor_implementation.hpp"

#include <yaml-cpp/yaml.h>

/**
 * Occupancy mapping voxel implementation.
 */
struct OFusion {

  /**
   * The voxel type used as the template parameter for se::Octree.
   */
  struct VoxelType {
    /**
     * The struct stored in each se::Octree voxel.
     */
    struct VoxelData {
      float  x; /**< The occupancy value in log-odds. */
      double y; /**< The timestamp of the last update. */

      bool operator==(const VoxelData& other) const;
      bool operator!=(const VoxelData& other) const;
    };

    static inline VoxelData invalid()     { return {0.f, 0.f}; }
    static inline VoxelData initData() { return {0.f, 0.f}; }

    using VoxelBlockType = se::VoxelBlockFull<OFusion::VoxelType>;

    using MemoryPoolType = se::PagedMemoryPool<OFusion::VoxelType>;
    template <typename ElemT>
    using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
  };

  using VoxelData      = OFusion::VoxelType::VoxelData;
  using OctreeType     = se::Octree<OFusion::VoxelType>;
  using VoxelBlockType = typename OFusion::VoxelType::VoxelBlockType;

  /**
   * No need to invert the normals when rendering an occupancy map.
   */
  static constexpr bool invert_normals = false;

  /**
   * The surface is considered to be where the log-odds occupancy probability
   * crosses this value.
   */
  static float surface_boundary;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be lower
   * than this value.
   */
  static float min_occupancy;

  /**
   * Stored occupancy probabilities in log-odds are clamped to never be greater
   * than this value.
   */
  static float max_occupancy;

  /**
   * The value of the time constant tau in equation (10) from \cite
   * VespaRAL18.
   */
  static float tau;

  static float sigma_min_factor;
  static float sigma_max_factor;

  static float sigma_min;
  static float sigma_max;

  /**
   * Grow rate factor of uncertainty
   */
  static float k_sigma;

  static std::string type() { return "ofusion"; }

  /**
   * Configure the OFusion parameters
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



  /**
   * Integrate a depth image into the map.
   */
  static void integrate(OctreeType&             map,
                        const se::Image<float>& depth_image,
                        const Eigen::Matrix4f&  T_CM,
                        const SensorImpl&       sensor,
                        const unsigned          frame);



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

