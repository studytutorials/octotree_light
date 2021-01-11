/*
 *
 * Copyright 2019 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * */

#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"

#include "se/node.hpp"
#include "se/octree.hpp"
#include "se/image/image.hpp"
#include "se/image_utils.hpp"
#include "se/filter.hpp"
#include "se/functors/for_each.hpp"



struct MultiresTSDFUpdate {

  using VoxelType      = MultiresTSDF::VoxelType;
  using VoxelData      = MultiresTSDF::VoxelType::VoxelData;
  using OctreeType     = se::Octree<MultiresTSDF::VoxelType>;
  using NodeType       = se::Node<MultiresTSDF::VoxelType>;
  using VoxelBlockType = typename MultiresTSDF::VoxelType::VoxelBlockType;

  MultiresTSDFUpdate(const OctreeType&       map,
                     const se::Image<float>& depth_image,
                     const Eigen::Matrix4f&  T_CM,
                     const SensorImpl        sensor,
                     const float             voxel_dim,
                     int frame = 0,
                     int weight = 1) :
      map_(map),
      depth_image_(depth_image),
      T_CM_(T_CM),
      sensor_(sensor),
      voxel_dim_(voxel_dim),
      sample_offset_frac_(map.sample_offset_frac_),
      frame_(frame),
      weight_(weight) {}

  const OctreeType& map_;
  const se::Image<float>& depth_image_;
  const Eigen::Matrix4f& T_CM_;
  const SensorImpl sensor_;
  const float voxel_dim_;
  const Eigen::Vector3f& sample_offset_frac_;
  int frame_;
  int weight_;

  /**
   * Update the subgrids of a voxel block starting from a given scale up
   * to a maximum scale.
   *
   * \param[in] block VoxelBlock to be updated
   * \param[in] scale scale from which propagate up voxel values
   */
  static void propagateUp(VoxelBlockType* block,
                          const int       scale) {
    const Eigen::Vector3i block_coord = block->coordinates();
    const int block_size = VoxelBlockType::size_li;
    for (int voxel_scale = scale; voxel_scale < se::math::log2_const(block_size); ++voxel_scale) {
      const int stride = 1 << (voxel_scale + 1);
      for (int z = 0; z < block_size; z += stride)
        for (int y = 0; y < block_size; y += stride)
          for (int x = 0; x < block_size; x += stride) {
            const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);

            float mean = 0;
            int sample_count = 0;
            float weight = 0;
            VoxelData voxel_data = block->data(voxel_coord, voxel_scale + 1);
            int frame = voxel_data.frame;
            for (int k = 0; k < stride; k += stride / 2) {
              for (int j = 0; j < stride; j += stride / 2) {
                for (int i = 0; i < stride; i += stride / 2) {
                  VoxelData child_data = block->data(voxel_coord + Eigen::Vector3i(i, j, k), voxel_scale);
                  if (child_data.y != 0) {
                    mean += child_data.x;
                    weight += child_data.y;
                    sample_count++;
                    frame = std::max(frame, child_data.frame); // sleutenegger
                  }
                }
              }
            }

            if (sample_count != 0) {
              mean /= sample_count;
              weight /= sample_count;
              voxel_data.x = mean;
              voxel_data.x_last = mean;
              voxel_data.y = ceil(weight);
            } else {
              voxel_data = VoxelType::initData();
            }
            voxel_data.delta_y = 0;
            voxel_data.frame = frame; // sleutenegger
            block->setData(voxel_coord, voxel_scale + 1, voxel_data);
          }
    }
  }



  static void propagateUp(NodeType*      node,
                          const int      voxel_depth,
                          const unsigned timestamp) {

    if (!node->parent()) {
      node->timestamp(timestamp);
      return;
    }

    float mean = 0;
    int sample_count = 0;
    float weight = 0;
    int frame = 0;
    for (int child_idx = 0; child_idx < 8; ++child_idx) {
      const VoxelData& child_data = node->childData(child_idx);
      if (child_data.y != 0) {
        mean += child_data.x;
        weight += child_data.y;
        sample_count++;
        frame = std::max(child_data.frame, frame); // sleutenegger: propagate newest frame update
      }
    }

    const unsigned int child_idx = se::child_idx(node->code(),
                                                 se::keyops::code(node->code()), voxel_depth);
    if (sample_count > 0) {
      VoxelData& node_data = node->parent()->childData(child_idx);
      mean /= sample_count;
      weight /= sample_count;
      node_data.x = mean;
      node_data.x_last = mean;
      node_data.y = ceil(weight);
      node_data.delta_y = 0;
      node_data.frame = std::max(frame, node_data.frame); // sleutenegger: propagate newest frame update
    }
    node->timestamp(timestamp);
  }



  /**
   * Update the subgrids of a voxel block starting from a given scale
   * down to the finest grid.
   *
   * \param[in] block VoxelBlock to be updated
   * \param[in] scale scale from which propagate down voxel values
   */
  static void propagateDown(const OctreeType& map,
                            VoxelBlockType*   block,
                            const int         scale,
                            const int         min_scale) {

    const Eigen::Vector3i block_coord = block->coordinates();
    const int block_size = VoxelBlockType::size_li;
    for (int voxel_scale = scale; voxel_scale > min_scale; --voxel_scale) {
      const int stride = 1 << voxel_scale;
      for (int z = 0; z < block_size; z += stride) {
        for (int y = 0; y < block_size; y += stride) {
          for (int x = 0; x < block_size; x += stride) {
            const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
            VoxelData parent_data = block->data(parent_coord, voxel_scale);
            float delta_x = parent_data.x - parent_data.x_last;
            const int half_stride = stride / 2;
            for (int k = 0; k < stride; k += half_stride) {
              for (int j = 0; j < stride; j += half_stride) {
                for (int i = 0; i < stride; i += half_stride) {
                  const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                  VoxelData voxel_data = block->data(voxel_coord, voxel_scale - 1);
                  if (voxel_data.y == 0) {
                    bool is_valid;
                    const Eigen::Vector3f voxel_sample_coord_f =
                        se::get_sample_coord(voxel_coord, stride, map.sample_offset_frac_);
                    voxel_data.x = se::math::clamp(map.interp(voxel_sample_coord_f,
                                                              VoxelType::selectNodeValue,
                                                              VoxelType::selectVoxelValue,
                                                              voxel_scale - 1, is_valid).first, -1.f, 1.f);
                    voxel_data.y = is_valid ? parent_data.y : 0;
                    voxel_data.x_last = voxel_data.x;
                    voxel_data.delta_y = 0;
                  } else {
                    voxel_data.x = std::max(voxel_data.x + delta_x, -1.f);
                    voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, MultiresTSDF::max_weight);
                    voxel_data.delta_y = parent_data.delta_y;
                  }
                  voxel_data.frame = parent_data.frame; // sleutenegger: inherit
                  block->setData(voxel_coord, voxel_scale - 1, voxel_data);
                }
              }
            }
            parent_data.x_last = parent_data.x;
            parent_data.delta_y = 0;
            //parent_data.frame = frame_;
            block->setData(parent_coord, voxel_scale, parent_data);
          }
        }
      }
    }
  }



  /**
   * Update a voxel block at a given scale by first propagating down the parent
   * values and then integrating the new measurement;
   */
  void propagateUpdate(VoxelBlockType* block,
                       const int       voxel_scale) {

    const int block_size = VoxelBlockType::size_li;
    const int parent_scale = voxel_scale + 1;
    const int parent_stride = 1 << parent_scale;
    const int voxel_stride = parent_stride >> 1;
    bool is_visible = false;

    const Eigen::Vector3i block_coord = block->coordinates();

    for (unsigned int z = 0; z < block_size; z += parent_stride) {
      for (unsigned int y = 0; y < block_size; y += parent_stride) {
        for (unsigned int x = 0; x < block_size; x += parent_stride) {
          const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
          VoxelData parent_data = block->data(parent_coord, parent_scale);
          float delta_x = parent_data.x - parent_data.x_last;
          for (int k = 0; k < parent_stride; k += voxel_stride) {
            for (int j = 0; j < parent_stride; j += voxel_stride) {
              for (int i = 0; i < parent_stride; i += voxel_stride) {
                const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                VoxelData voxel_data = block->data(voxel_coord, voxel_scale);
                const Eigen::Vector3f voxel_sample_coord_f =
                    se::get_sample_coord(voxel_coord, voxel_stride, sample_offset_frac_);
                if (voxel_data.y == 0) {
                  bool is_valid;
                  voxel_data.x = se::math::clamp(map_.interp(voxel_sample_coord_f,
                                                             VoxelType::selectNodeValue,
                                                             VoxelType::selectVoxelValue,
                                                             voxel_scale + 1, is_valid).first, -1.f, 1.f);
                  voxel_data.y = is_valid ? parent_data.y : 0;
                  voxel_data.x_last = voxel_data.x;
                  voxel_data.delta_y = 0;
                } else {
                  voxel_data.x = se::math::clamp(voxel_data.x + delta_x, -1.f, 1.f);
                  voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, MultiresTSDF::max_weight);
                  voxel_data.delta_y = parent_data.delta_y;
                }

                const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous()).head(3);

                // sleutenegger: don't update if beyond MultiresTSDF::stop_weight
                if(MultiresTSDF::stop_weight >= 0) {
                  if(voxel_data.y > MultiresTSDF::stop_weight) {
                    continue;
                  }
                }

                // Don't update the point if the sample point is behind the far plane
                if (point_C.norm() > sensor_.farDist(point_C)) {
                  continue;
                }

                float depth_value(0);
                if (!sensor_.projectToPixelValue(point_C, depth_image_, depth_value,
                    [](float depth_value){ return depth_value > 0; })) {
                  continue;
                }

                is_visible = true;

                // Update the TSDF
                const float m = sensor_.measurementFromPoint(point_C);
                const float sdf_value = (depth_value - m) / m * point_C.norm();
                if (sdf_value > -MultiresTSDF::mu * (1 << voxel_scale)) {
                  const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
                  voxel_data.x = se::math::clamp(
                      (static_cast<float>(voxel_data.y) * voxel_data.x + weight_ * tsdf_value) /
                      (static_cast<float>(voxel_data.y + weight_)), -1.f, 1.f);
                  voxel_data.y = fminf(voxel_data.y + 1, MultiresTSDF::max_weight);
                  voxel_data.delta_y+=weight_;
                }
                voxel_data.frame = frame_;
                block->setData(voxel_coord, voxel_scale, voxel_data);
              }
            }
          }
          parent_data.x_last = parent_data.x;
          parent_data.delta_y = 0;
          parent_data.frame = frame_;
          block->setData(parent_coord, parent_scale, parent_data);
        }
      }
    }
    block->current_scale(voxel_scale);
    block->active(is_visible);
  }



  void operator()(VoxelBlockType* block) {

    constexpr int block_size = VoxelBlockType::size_li;
    const Eigen::Vector3i block_coord = block->coordinates();
    const Eigen::Vector3f block_centre_coord_f =
        se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
    const Eigen::Vector3f block_centre_point_C = (T_CM_ * (voxel_dim_ * block_centre_coord_f).homogeneous()).head(3);
    const int last_scale = block->current_scale();

    const int scale = std::max(sensor_.computeIntegrationScale(
        block_centre_point_C, voxel_dim_, last_scale, block->min_scale(), map_.maxBlockScale()), last_scale - 1);
    block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));
    if (last_scale > scale) {
      propagateUpdate(block, scale);
      return;
    }
    bool is_visible = false;
    block->current_scale(scale);
    const int stride = 1 << scale;
    for (unsigned int z = 0; z < block_size; z += stride) {
      for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
        for (unsigned int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3f voxel_sample_coord_f =
              se::get_sample_coord(voxel_coord, stride, sample_offset_frac_);
          const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous()).head(3);

          // Don't update the point if the sample point is behind the far plane
          if (point_C.norm() > sensor_.farDist(point_C)) {
            continue;
          }
          float depth_value(0);
          if (!sensor_.projectToPixelValue(point_C, depth_image_, depth_value,
              [&](float depth_value){ return depth_value >= sensor_.near_plane; })) {
            continue;
          }

          is_visible = true;

          // Update the TSDF
          const float m = sensor_.measurementFromPoint(point_C);
          const float sdf_value = (depth_value - m) / m * point_C.norm();
          if (sdf_value > -MultiresTSDF::mu  * (1 << scale)) {
            const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
            VoxelData voxel_data = block->data(voxel_coord, scale);
            // sleutenegger: don't update if beyond MultiresTSDF::stop_weight
            if(MultiresTSDF::stop_weight >= 0) {
              if(voxel_data.y > MultiresTSDF::stop_weight) {
                continue;
              }
            }
            voxel_data.x = se::math::clamp(
                (static_cast<float>(voxel_data.y) * voxel_data.x + weight_ * tsdf_value) /
                (static_cast<float>(voxel_data.y + weight_)),
                -1.f, 1.f);
            voxel_data.y = fminf(voxel_data.y + 1, MultiresTSDF::max_weight);
            voxel_data.delta_y+=weight_;
            voxel_data.frame = frame_;
            block->setData(voxel_coord, scale, voxel_data);
          }
        }
      }
    }
    propagateUp(block, scale);
    block->active(is_visible);
  }
};

struct MultiresTSDFUpdateRangeMeasurements {

  using VoxelType      = MultiresTSDF::VoxelType;
  using VoxelData      = MultiresTSDF::VoxelType::VoxelData;
  using OctreeType     = se::Octree<MultiresTSDF::VoxelType>;
  using NodeType       = se::Node<MultiresTSDF::VoxelType>;
  using VoxelBlockType = typename MultiresTSDF::VoxelType::VoxelBlockType;

  MultiresTSDFUpdateRangeMeasurements(const OctreeType&       map,
                     const std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>>& ranges,
                     const Eigen::Matrix4f&  T_CM,
                     const float             voxel_dim,
                     int frame = 0) :
      map_(map),
      ranges_(ranges),
      T_CM_(T_CM),
      voxel_dim_(voxel_dim),
      sample_offset_frac_(map.sample_offset_frac_),
      frame_(frame) {
    for(const se::RangeMeasurement & range : ranges_) {
      max_range_ = std::max(range.range.norm(), max_range_);
      min_range_ = std::min(range.range.norm(), min_range_);
    }
  }

  const OctreeType& map_;
  const std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>>& ranges_;
  const Eigen::Matrix4f& T_CM_;
  const float voxel_dim_;
  const Eigen::Vector3f& sample_offset_frac_;
  float max_range_ = 0.0f;
  float min_range_ = std::numeric_limits<float>::max();
  int frame_;

  /**
   * Update the subgrids of a voxel block starting from a given scale up
   * to a maximum scale.
   *
   * \param[in] block VoxelBlock to be updated
   * \param[in] scale scale from which propagate up voxel values
   */
  static void propagateUp(VoxelBlockType* block,
                          const int       scale) {
    MultiresTSDFUpdate::propagateUp(block, scale);
  }


  static void propagateUp(NodeType*      node,
                          const int      voxel_depth,
                          const unsigned timestamp) {

    MultiresTSDFUpdate::propagateUp(node, voxel_depth, timestamp);
  }


  static void propagateDown(const OctreeType& map,
                            VoxelBlockType*   block,
                            const int         scale,
                            const int         min_scale) {

    MultiresTSDFUpdate::propagateDown(map, block, scale, min_scale);
  }

  /**
   * Update a voxel block at a given scale by first propagating down the parent
   * values and then integrating the new measurement;
   */
  void propagateUpdate(VoxelBlockType* block,
                       const int       voxel_scale) {

    const int block_size = VoxelBlockType::size_li;
    const int parent_scale = voxel_scale + 1;
    const int parent_stride = 1 << parent_scale;
    const int voxel_stride = parent_stride >> 1;
    bool is_visible = false;

    const Eigen::Vector3i block_coord = block->coordinates();

    // sleutene: early abort if whole block too far away
    const Eigen::Vector3f voxel_centre_coord_f =
                    se::get_sample_coord(block_coord, block_size, sample_offset_frac_);
    const Eigen::Vector3f voxel_centre_coord_C = (T_CM_ * (voxel_dim_ * voxel_centre_coord_f).homogeneous()).head(3);
    const float block_angle = atan(voxel_dim_*block_size*sqrt(3.0f)/2.0f/voxel_centre_coord_C.norm());
    std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>> ranges;
    for(const se::RangeMeasurement & range : ranges_) {
      if(voxel_centre_coord_C.normalized().dot(range.range.normalized())>cos(range.beamDivergence + block_angle)) {
        ranges.push_back(range); // only consider active ones later
      }
    }

    for (unsigned int z = 0; z < block_size; z += parent_stride) {
      for (unsigned int y = 0; y < block_size; y += parent_stride) {
        for (unsigned int x = 0; x < block_size; x += parent_stride) {
          const Eigen::Vector3i parent_coord = block_coord + Eigen::Vector3i(x, y, z);
          VoxelData parent_data = block->data(parent_coord, parent_scale);
          float delta_x = parent_data.x - parent_data.x_last;
          for (int k = 0; k < parent_stride; k += voxel_stride) {
            for (int j = 0; j < parent_stride; j += voxel_stride) {
              for (int i = 0; i < parent_stride; i += voxel_stride) {
                const Eigen::Vector3i voxel_coord = parent_coord + Eigen::Vector3i(i, j, k);
                VoxelData voxel_data = block->data(voxel_coord, voxel_scale);
                const Eigen::Vector3f voxel_sample_coord_f =
                    se::get_sample_coord(voxel_coord, voxel_stride, sample_offset_frac_);
                if (voxel_data.y == 0) {
                  bool is_valid;
                  voxel_data.x = se::math::clamp(map_.interp(voxel_sample_coord_f,
                                                             VoxelType::selectNodeValue,
                                                             VoxelType::selectVoxelValue,
                                                             voxel_scale + 1, is_valid).first, -1.f, 1.f);
                  voxel_data.y = is_valid ? parent_data.y : 0;
                  voxel_data.x_last = voxel_data.x;
                  voxel_data.delta_y = 0;
                } else {
                  voxel_data.x = se::math::clamp(voxel_data.x + delta_x, -1.f, 1.f);
                  voxel_data.y = fminf(voxel_data.y + parent_data.delta_y, MultiresTSDF::max_weight);
                  voxel_data.delta_y = parent_data.delta_y;
                }

                // sleutenegger: don't update if beyond MultiresTSDF::stop_weight
                if(MultiresTSDF::stop_weight >= 0) {
                  if(voxel_data.y > MultiresTSDF::stop_weight) {
                    continue;
                  }
                }

                const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous()).head(3);

                // find corresponding range measurement.
                /// \todo sleutene: this will be slow. Think about different algorithms here
                /// (e.g. pre-computed 2d spherical association map).
                for(const se::RangeMeasurement & range : ranges) {
                  if(point_C.normalized().dot(range.range.normalized())>cos(range.beamDivergence)) {
                    // inside the cone.
                    is_visible = true;

                    // compute SDF value by projecting distance vector onto range measurement direction.
                    const float sdf_value = (range.range - point_C).dot(range.range.normalized());

                    // Update the TSDF
                    if (sdf_value > -MultiresTSDF::mu * (1 << voxel_scale)) {
                      const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
                      voxel_data.x = se::math::clamp(
                          (static_cast<float>(voxel_data.y) * voxel_data.x + static_cast<float>(range.weight) * tsdf_value) /
                          (static_cast<float>(voxel_data.y) + static_cast<float>(range.weight)), -1.f, 1.f);
                      voxel_data.y = fminf(voxel_data.y + range.weight, MultiresTSDF::max_weight);
                      voxel_data.delta_y+=range.weight;
                    }
                    voxel_data.frame = frame_;
                    block->setData(voxel_coord, voxel_scale, voxel_data);
                  }
                }
              }
            }
          }
          parent_data.x_last = parent_data.x;
          parent_data.delta_y = 0;
          parent_data.frame = frame_;
          block->setData(parent_coord, parent_scale, parent_data);
        }
      }
    }
    block->current_scale(voxel_scale);
    block->active(is_visible);
  }



  void operator()(VoxelBlockType* block) {

    constexpr int block_size = VoxelBlockType::size_li;
    const Eigen::Vector3i block_coord = block->coordinates();
    const Eigen::Vector3f block_centre_coord_f =
        se::get_sample_coord(block_coord, block_size, Eigen::Vector3f::Constant(0.5f));
    const Eigen::Vector3f block_centre_point_C = (T_CM_ * (voxel_dim_ * block_centre_coord_f).homogeneous()).head(3);
    const int last_scale = block->current_scale();

    // sleutene: some more early aborting
    if(block_centre_point_C.norm() > max_range_ + voxel_dim_*block_size*sqrt(3.0f)/2.0) {
      return;
    }
    if(block_centre_point_C.norm() < min_range_ - voxel_dim_*block_size*sqrt(3.0f)/2.0) {
      return;
    }

    // compute scale
    int r=0;
    float max_cos_angle = 0.0;
    int max_r = -1;
    for(const se::RangeMeasurement & range : ranges_) {
      const float cos_angle = block_centre_point_C.normalized().dot(range.range.normalized());
      if(cos_angle>max_cos_angle) {
        max_r = r;
        max_cos_angle = cos_angle;
      }
      ++r;
    }
    const int ideal_scale = std::floor(std::log2((block_centre_point_C.norm() * sin(ranges_[max_r].beamDivergence)) / voxel_dim_ / 1.5));
    const int computed_scale = std::min(std::max(0, ideal_scale), map_.maxBlockScale());
    const int scale = std::max(computed_scale, last_scale - 1);
    block->min_scale(block->min_scale() < 0 ? scale : std::min(block->min_scale(), scale));

    if (last_scale > scale) {
      propagateUpdate(block, scale);
      return;
    }

    // sleutene: early abort if whole block too far away
    const float block_angle = atan(block_size*voxel_dim_*sqrt(3.0f)/2.0f/block_centre_point_C.norm());
    std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>> ranges;
    for(const se::RangeMeasurement & range : ranges_) {
      if(block_centre_point_C.normalized().dot(range.range.normalized())>cos(range.beamDivergence + block_angle)) {
        ranges.push_back(range); // only consider active ones later
      }
    }

    bool is_visible = false;
    block->current_scale(scale);
    const int stride = 1 << scale;
    for (unsigned int z = 0; z < block_size; z += stride) {
      for (unsigned int y = 0; y < block_size; y += stride) {
#pragma omp simd
        for (unsigned int x = 0; x < block_size; x += stride) {
          const Eigen::Vector3i voxel_coord = block_coord + Eigen::Vector3i(x, y, z);
          const Eigen::Vector3f voxel_sample_coord_f =
              se::get_sample_coord(voxel_coord, stride, sample_offset_frac_);
          const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim_ * voxel_sample_coord_f).homogeneous()).head(3);

          // find corresponding range measurement.
          /// \todo sleutene: this will be slow. Think about different algorithms here
          /// (e.g. pre-computed 2d spherical association map).
          for(const se::RangeMeasurement & range : ranges) {
            if(point_C.normalized().dot(range.range.normalized())>cos(range.beamDivergence)) {
              // inside the cone.
              is_visible = true;

              // compute SDF value by projecting distance vector onto range measurement direction.
                  const float sdf_value = (range.range - point_C).dot(range.range.normalized());
              if (sdf_value > -MultiresTSDF::mu  * (1 << scale)) {
                const float tsdf_value = fminf(1.f, sdf_value / MultiresTSDF::mu);
                VoxelData voxel_data = block->data(voxel_coord, scale);
                // sleutenegger: don't update if beyond MultiresTSDF::stop_weight
                if(MultiresTSDF::stop_weight >= 0) {
                  if(voxel_data.y > MultiresTSDF::stop_weight) {
                    continue;
                  }
                }
                voxel_data.x = se::math::clamp(
                    (static_cast<float>(voxel_data.y) * voxel_data.x + static_cast<float>(range.weight)*tsdf_value) /
                    (static_cast<float>(voxel_data.y) + static_cast<float>(range.weight)),
                    -1.f, 1.f);
                voxel_data.y = fminf(voxel_data.y + range.weight, MultiresTSDF::max_weight);
                voxel_data.delta_y+=range.weight;
                voxel_data.frame = frame_;
                block->setData(voxel_coord, scale, voxel_data);
              }
            }
          }
        }
      }
    }
    propagateUp(block, scale);
    block->active(is_visible);
  }
};

void MultiresTSDF::integrate(OctreeType&             map,
                             const se::Image<float>& depth_image,
                             const Eigen::Matrix4f&  T_CM,
                             const SensorImpl&       sensor,
                             const unsigned          frame,
                             int weight) {

  using namespace std::placeholders;

  /* Retrieve the active list */
  std::vector<VoxelBlockType *> active_list;
  auto& block_buffer = map.pool().blockBuffer();

  /* Predicates definition */
  const float voxel_dim = map.dim() / map.size();
  auto in_frustum_predicate =
  std::bind(se::algorithms::in_frustum<VoxelBlockType>,
      std::placeholders::_1, voxel_dim, T_CM, sensor);
  auto is_active_predicate = [](const VoxelBlockType* block) {
    return block->active();
  };
  se::algorithms::filter(active_list, block_buffer, is_active_predicate,
                         in_frustum_predicate);

  std::deque<se::Node<VoxelType> *> node_queue;
  struct MultiresTSDFUpdate block_update_funct(
      map, depth_image, T_CM, sensor, voxel_dim, frame, weight);
  se::functor::internal::parallel_for_each(active_list, block_update_funct);

  for (const auto& block : active_list) {
    if (block->parent()) {
      node_queue.push_back(block->parent());
    }
  }

  while (!node_queue.empty()) {
    se::Node<VoxelType>* node = node_queue.front();
    node_queue.pop_front();
    if (node->timestamp() == frame) {
      continue;
    }
    block_update_funct.propagateUp(node, map.voxelDepth(), frame);
    if (node->parent()) {
      node_queue.push_back(node->parent());
    }
  }
}

void MultiresTSDF::integrateRangeMeasurements(
    OctreeType& map, const std::vector<se::RangeMeasurement, Eigen::aligned_allocator<se::RangeMeasurement>>& ranges,
    const Eigen::Matrix4f& T_CM, const unsigned frame) {
  /* Retrieve the active list */
  std::vector<VoxelBlockType *> active_list;
  auto& block_buffer = map.pool().blockBuffer();

  /* Predicates definition */
  const float voxel_dim = map.dim() / map.size();
  auto is_active_predicate = [](const VoxelBlockType* block) {
    return block->active();
  };
  se::algorithms::filter(active_list, block_buffer, is_active_predicate);

  std::deque<se::Node<VoxelType> *> node_queue;
  struct MultiresTSDFUpdateRangeMeasurements block_update_funct(
      map, ranges, T_CM, voxel_dim, frame);
  se::functor::internal::parallel_for_each(active_list, block_update_funct);

  for (const auto& block : active_list) {
    if (block->parent()) {
      node_queue.push_back(block->parent());
    }
  }

  while (!node_queue.empty()) {
    se::Node<VoxelType>* node = node_queue.front();
    node_queue.pop_front();
    if (node->timestamp() == frame) {
      continue;
    }
    block_update_funct.propagateUp(node, map.voxelDepth(), frame);
    if (node->parent()) {
      node_queue.push_back(node->parent());
    }
  }
}
