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

#ifndef PROJECTIVE_FUNCTOR_HPP
#define PROJECTIVE_FUNCTOR_HPP
#include <functional>
#include <vector>

#include "se/utils/math_utils.h"
#include "filter.hpp"
#include "se/node.hpp"
#include "se/functors/data_handler.hpp"
#include "se/sensor_implementation.hpp"

namespace se {
namespace functor {
  template <typename FieldType, template <typename FieldT> class OctreeT,
            typename UpdateF>
  class projective_functor {

    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      projective_functor(OctreeT<FieldType>&    octree,
                         UpdateF&               funct,
                         const Eigen::Matrix4f& T_CM,
                         const SensorImpl       sensor,
                         const Eigen::Vector3f& sample_offset_frac,
                         const Eigen::Vector2i& image_res) :
        octree_(octree),
        funct_(funct),
        T_CM_(T_CM),
        sensor_(sensor),
        sample_offset_frac_(sample_offset_frac),
        image_res_(image_res) {
      }

      /*! \brief Get all the blocks that are active or inside the camera
       * frustum. The blocks are stored in projective_functor::active_list_.
       */
      void build_active_list() {
        using namespace std::placeholders;
        /* Retrieve the active list */
        const typename FieldType::template MemoryBufferType<se::VoxelBlock<FieldType>>& block_buffer = octree_.pool().blockBuffer();

        /* Predicates definition */
        const float voxel_dim = octree_.dim() / octree_.size();
        auto in_frustum_predicate =
          std::bind(algorithms::in_frustum<se::VoxelBlock<FieldType>>,
              std::placeholders::_1, voxel_dim, T_CM_, sensor_);
        auto is_active_predicate = [](const se::VoxelBlock<FieldType>* block) {
          return block->active();
        };

        /* Get all the blocks that are active or inside the camera frustum. */
        algorithms::filter(active_list_, block_buffer, is_active_predicate,
            in_frustum_predicate);
      }

      void update_block(se::VoxelBlock<FieldType>* block,
                        const float                voxel_dim) {
        /* Is this the VoxelBlock centre? */
        const Eigen::Vector3i block_coord = block->coordinates();
        bool is_visible = false;
        block->current_scale(0);

        /* Iterate over each voxel in the VoxelBlock. */
        const unsigned int block_size = se::VoxelBlock<FieldType>::size;
        const unsigned int x_last = block_coord.x() + block_size;
        const unsigned int y_last = block_coord.y() + block_size;
        const unsigned int z_last = block_coord.z() + block_size;

        for (unsigned int z = block_coord.z(); z < z_last; ++z) {
          for (unsigned int y = block_coord.y(); y < y_last; ++y) {
#pragma omp simd
            for (unsigned int x = block_coord.x(); x < x_last; ++x) {
              const Eigen::Vector3i voxel_coord = Eigen::Vector3i(x, y, z);
              const Eigen::Vector3f point_C = (T_CM_ * (voxel_dim * (voxel_coord.cast<float>() +
                  sample_offset_frac_)).homogeneous()).head(3);

              Eigen::Vector2f pixel_f;
              if (sensor_.model.project(point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
                continue;
              }

              is_visible = true;

              /* Update the voxel. */
              VoxelBlockHandler<FieldType> handler = {block, voxel_coord};
              funct_(handler, voxel_coord, point_C, pixel_f);
            }
          }
        }
        block->active(is_visible);
      }

      void update_node(se::Node<FieldType>* node,
                       const float          voxel_dim) {
        const Eigen::Vector3i node_coord = Eigen::Vector3i(unpack_morton(node->code_));


        /* Iterate over the Node children. */
#pragma omp simd
        for(int child_idx = 0; child_idx < 8; ++child_idx) {
          const Eigen::Vector3i rel_step = node->size_ / 2 *
              Eigen::Vector3i((child_idx & 1) > 0, (child_idx & 2) > 0, (child_idx & 4) > 0); // TODO: Offset needs to be discussed
          const Eigen::Vector3i child_coord = node_coord + rel_step;
          const Eigen::Vector3f child_point_C = (T_CM_ * (voxel_dim * (child_coord.cast<float>() + node->size_ *
              sample_offset_frac_)).homogeneous()).head(3);
          Eigen::Vector2f pixel_f;
          if (sensor_.model.project(child_point_C, &pixel_f) != srl::projection::ProjectionStatus::Successful) {
            continue;
          }

          /* Update the child Node. */
          NodeHandler<FieldType> handler = {node, child_idx};
          funct_(handler, child_coord, child_point_C, pixel_f);
        }
      }

      void apply() {

        const float voxel_dim = octree_.dim() / octree_.size();

        /* Update the leaf Octree nodes (VoxelBlock). */
        build_active_list();
#pragma omp parallel for
        for (unsigned int i = 0; i < active_list_.size(); ++i) {
          update_block(active_list_[i], voxel_dim);
        }
        active_list_.clear();

        /* Update the intermediate Octree nodes (Node). */
        typename FieldType::template MemoryBufferType<se::Node<FieldType>>& node_buffer = octree_.pool().nodeBuffer();
#pragma omp parallel for
          for (unsigned int i = 0; i < node_buffer.size(); ++i) {
            update_node(node_buffer[i], voxel_dim);
         }
      }

    private:
      OctreeT<FieldType>& octree_;
      UpdateF& funct_;
      const Eigen::Matrix4f& T_CM_;
      const SensorImpl sensor_;
      const Eigen::Vector3f sample_offset_frac_;
      const Eigen::Vector2i image_res_;
      std::vector<se::VoxelBlock<FieldType>*> active_list_;
  };

  /*! \brief Create a projective_functor and call projective_functor::apply.
   */
  template <typename FieldType, template <typename FieldT> class OctreeT,
            typename UpdateF>
  void projective_octree(OctreeT<FieldType>&    octree,
                         const Eigen::Vector3f& sample_offset_frac,
                         const Eigen::Matrix4f& T_CM,
                         const SensorImpl&      sensor,
                         const Eigen::Vector2i& image_res,
                         UpdateF&               funct) {

    projective_functor<FieldType, OctreeT, UpdateF>
      it(octree, funct, T_CM, sensor, sample_offset_frac, image_res);
    it.apply();
  }
}
}
#endif
