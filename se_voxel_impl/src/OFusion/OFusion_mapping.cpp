/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
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

#include "se/voxel_implementations/OFusion/OFusion.hpp"

#include <algorithm>

#include "se/node.hpp"
#include "se/projective_functor.hpp"
#include "se/image/image.hpp"
#include "se/image_utils.hpp"
#include "OFusion_bspline_lookup.cc"



/**
 * Struct to hold the data and perform the update of the map from a single
 * depth frame.
 */
struct OFusionUpdate {
  const se::Image<float>& depth_image;
  const SensorImpl&       sensor;
  float                   timestamp;
  float                   voxel_dim;
  bool                    is_visible;



  OFusionUpdate(const se::Image<float>& depth_image,
                const SensorImpl&       sensor,
                float                   timestamp,
                float                   voxel_dim) :
      depth_image(depth_image),
      sensor(sensor),
      timestamp(timestamp),
      voxel_dim(voxel_dim) {};

  void reset() {
    is_visible = false;
  }



  template <typename FieldType,
      template <typename FieldT> class VoxelBlockT>
  void operator()(VoxelBlockT<FieldType>* block) {
    block->active(is_visible);
  }



  /**
   * Compute the value of the q_cdf spline using a lookup table. This implements
   * equation (7) from \cite VespaRAL18.
   *
   * \param[in] t Where to compute the value of the spline at.
   * \return The value of the spline.
   */
  inline float ofusion_bspline_memoized(float t) {
    float value = 0.f;
    constexpr float inverse_range = 1.f / 6.f;
    if (t >= -3.0f && t <= 3.0f) {
      const unsigned int idx
          = ((t + 3.f) * inverse_range) * (bspline_num_samples - 1) + 0.5f;
      return bspline_lookup[idx];
    } else if (t > 3.f) {
      value = 1.f;
    }
    return value;
  }



  /**
   * Compute the occupancy probability along the ray from the camera. This
   * implements equation (6) from \cite VespaRAL18.
   *
   * \param[in] val The point on the ray at which the occupancy probability is
   * computed. The point is expressed using the ray parametric equation.
   * \return The occupancy probability.
   */
  inline float ofusion_H(const float val) {
    const float Q_1 = ofusion_bspline_memoized(val);
    const float Q_2 = ofusion_bspline_memoized(val - 3);
    return Q_1 - Q_2 * 0.5f;
  }



  /**
   * Perform a log-odds update of the occupancy probability. This implements
   * equations (8) and (9) from \cite VespaRAL18.
   */
  inline float ofusion_update_logs(const float prior,
                                   const float sample) {
    return (prior + log2(sample / (1.f - sample)));
  }



  /**
   * Weight the occupancy by the time since the last update, acting as a
   * forgetting factor. This implements equation (10) from \cite VespaRAL18.
   */
  inline float ofusion_apply_window(const float occupancy,
                                    const float,
                                    const float delta_t,
                                    const float tau) {
    float fraction = 1.f / (1.f + (delta_t / tau));
    fraction = std::max(0.5f, fraction);
    return occupancy * fraction;
  }



  template <typename DataHandlerT>
  void operator()(DataHandlerT&          handler,
                  const Eigen::Vector3i&,
                  const Eigen::Vector3f& point_C) {
    // Don't update the point if the sample point is behind the far plane
    if (point_C.norm() > sensor.farDist(point_C)) {
      return;
    }

    // Don't update the point if the depth value is closer than the near plane
    float depth_value(0);
    if (!sensor.projectToPixelValue(point_C, depth_image, depth_value,
        [&](float depth_value){ return depth_value >= sensor.near_plane; })) {
      return;
    }

    is_visible = true;

    // Compute the occupancy probability for the current measurement.
    const float m = sensor.measurementFromPoint(point_C);
    const float diff = (m - depth_value);
    const float sigma = se::math::clamp(OFusion::k_sigma * se::math::sq(m), OFusion::sigma_min, OFusion::sigma_max);
    float sample = ofusion_H(diff / sigma);
    if (sample == 0.5f) {
      return;
    }
    sample = se::math::clamp(sample, 0.03f, 0.97f);

    auto data = handler.get();

    // Update the occupancy probability.
    const double delta_t = timestamp - data.y;
    data.x = ofusion_apply_window(data.x, OFusion::surface_boundary, delta_t, OFusion::tau);
    data.x = ofusion_update_logs(data.x, sample);
    data.x = se::math::clamp(data.x, OFusion::min_occupancy, OFusion::max_occupancy);
    data.y = timestamp;

    handler.set(data);
  }
};



void OFusion::integrate(OctreeType&             map,
                        const se::Image<float>& depth_image,
                        const Eigen::Matrix4f&  T_CM,
                        const SensorImpl&       sensor,
                        const unsigned          frame) {

  const float timestamp = (1.f / 30.f) * frame;
  const float voxel_dim =  map.dim() / map.size();
  const Eigen::Vector2i depth_image_res(depth_image.width(), depth_image.height());

  struct OFusionUpdate funct(depth_image, sensor, timestamp, voxel_dim);

  se::functor::projective_octree(map, map.sample_offset_frac_, T_CM, sensor, depth_image_res, funct);
}

