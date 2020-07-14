/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.


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

#ifndef __RENDERING_HPP
#define __RENDERING_HPP

#include <cstdint>

#include "se/utils/math_utils.h"
#include "se/octree.hpp"
#include "se/commons.h"
#include "lodepng.h"
#include "se/timings.h"
#include "se/image/image.hpp"
#include "se/tracking.hpp"

#include "se/voxel_implementations.hpp"
#include "se/sensor_implementation.hpp"



namespace se {
  namespace internal {
    static se::Image<int> scale_image(640, 480);
    static std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
      color_map =
      {
        {102, 194, 165},
        {252, 141, 98},
        {141, 160, 203},
        {231, 138, 195},
        {166, 216, 84},
        {255, 217, 47},
        {229, 196, 148},
        {179, 179, 179},
      };
  }
}



template<typename T>
void raycastKernel(const se::Octree<typename T::VoxelType>& map,
                   se::Image<Eigen::Vector3f>&              surface_point_cloud_M,
                   se::Image<Eigen::Vector3f>&              surface_normals_M,
                   const Eigen::Matrix4f&                   raycast_T_MC,
                   const SensorImpl&                        sensor) {

  TICK();
#pragma omp parallel for
  for (int y = 0; y < surface_point_cloud_M.height(); y++) {
#pragma omp simd
    for (int x = 0; x < surface_point_cloud_M.width(); x++) {
      Eigen::Vector4f surface_intersection_M;

      const Eigen::Vector2i pixel(x, y);
      const Eigen::Vector2f pixel_f = pixel.cast<float>();
      Eigen::Vector3f ray_dir_C;
      sensor.model.backProject(pixel_f, &ray_dir_C);
      const Eigen::Vector3f ray_dir_M = (se::math::to_rotation(raycast_T_MC) * ray_dir_C.normalized()).head(3);
      const Eigen::Vector3f t_MC = se::math::to_translation(raycast_T_MC);

      surface_intersection_M = T::raycast(map, t_MC, ray_dir_M, sensor.nearDist(ray_dir_C), sensor.farDist(ray_dir_C));
      if (surface_intersection_M.w() >= 0.f) {
        surface_point_cloud_M[x + y * surface_point_cloud_M.width()] = surface_intersection_M.head<3>();
        Eigen::Vector3f surface_normal = map.gradAtPoint(surface_intersection_M.head<3>(),
            [](const auto& data){ return data.x; },
            static_cast<int>(surface_intersection_M.w() + 0.5f));
        se::internal::scale_image(x, y) = static_cast<int>(surface_intersection_M.w());
        if (surface_normal.norm() == 0.f) {
          surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] = Eigen::Vector3f(INVALID, 0.f, 0.f);
        } else {
          // Invert surface normals for TSDF representations.
          surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] = T::invert_normals
              ? (-1.f * surface_normal).normalized()
              : surface_normal.normalized();
        }
      } else {
        surface_point_cloud_M[pixel.x() + pixel.y() * surface_point_cloud_M.width()] = Eigen::Vector3f::Zero();
        surface_normals_M[pixel.x() + pixel.y() * surface_normals_M.width()] = Eigen::Vector3f(INVALID, 0.f, 0.f);
      }
    }
  }
  TOCK("raycastKernel", surface_point_cloud_M.width() * surface_point_cloud_M.height());
}



void renderRGBAKernel(uint32_t*                  output_RGBA_image_data,
                      const Eigen::Vector2i&     output_RGBA_image_res,
                      const se::Image<uint32_t>& input_RGBA_image);



void renderDepthKernel(uint32_t*              depth_RGBA_image_data,
                       float*                 depth_image_data,
                       const Eigen::Vector2i& depth_RGBA_image_res,
                       const float            near_plane,
                       const float            far_plane);



void renderTrackKernel(uint32_t*              tracking_RGBA_image_data,
                       const TrackData*       tracking_result_data,
                       const Eigen::Vector2i& tracking_RGBA_image_res);



template <typename T>
void renderVolumeKernel(uint32_t*                         volume_RGBA_image_data,
                        const Eigen::Vector2i&            volume_RGBA_image_res,
                        const Eigen::Vector3f&            light_M,
                        const Eigen::Vector3f&            ambient_M,
                        const se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                        const se::Image<Eigen::Vector3f>& surface_normals_M) {
  TICK();
#pragma omp parallel for
  for (int y = 0; y < volume_RGBA_image_res.y(); y++) {
#pragma omp simd
    for (int x = 0; x < volume_RGBA_image_res.x(); x++) {

      const size_t pixel_idx = x + volume_RGBA_image_res.x() * y;

      const Eigen::Vector3f surface_point_M = surface_point_cloud_M[pixel_idx];
      const Eigen::Vector3f surface_normal_M = surface_normals_M[pixel_idx];

      if (surface_normal_M.x() != INVALID && surface_normal_M.norm() > 0.f) {
        const Eigen::Vector3f diff = (surface_point_M - light_M).normalized();
        const Eigen::Vector3f dir
            = Eigen::Vector3f::Constant(fmaxf(surface_normal_M.normalized().dot(diff), 0.f));
        Eigen::Vector3f col = dir + ambient_M;
        se::math::clamp(col, Eigen::Vector3f::Zero(), Eigen::Vector3f::Ones());
        col = col.cwiseProduct(se::internal::color_map[se::internal::scale_image(x, y)]);
        volume_RGBA_image_data[pixel_idx] = se::pack_rgba(col.x(), col.y(), col.z(), 0xFF);
      } else {
        volume_RGBA_image_data[pixel_idx] = 0xFF000000;
      }
    }
  }
  TOCK("renderVolumeKernel", volume_RGBA_image_res.prod());
}



inline void printNormals(const se::Image<Eigen::Vector3f>& normals,
                         const char*                       filename);



// Find ALL the intersection along a ray till the far_plane.
template <typename T>
void raycast_full(
    const se::Octree<typename T::VoxelType>&                                 map,
    std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>& surface_points_M,
    const Eigen::Vector3f&                                                   ray_origin_M,
    const Eigen::Vector3f&                                                   ray_dir_M,
    const float                                                              far_plane,
    const float                                                              step,
    const float                                                              large_step) {

  float t = 0;
  float step_size = large_step;
  float f_t = map.interpAtPoint(ray_origin_M + ray_dir_M * t, [](const auto& data){ return data.x;}).first;
  t += step;
  float f_tt = 1.f;

  for (; t < far_plane; t += step_size) {
    f_tt = map.interpAtPoint(ray_origin_M + ray_dir_M * t, [](const auto& data){ return data.x;}).first;
    if (f_tt < 0.f && f_t > 0.f && std::abs(f_tt - f_t) < 0.5f) {     // got it, jump out of inner loop
      const auto data_t  = map.getFineAtPoint(ray_origin_M + ray_dir_M * (t - step_size));
      const auto data_tt = map.getFineAtPoint(ray_origin_M + ray_dir_M * t);
      if (f_t == 1.0 || f_tt == 1.0 || data_t.y == 0 || data_tt.y == 0 ) {
        f_t = f_tt;
        continue;
      }
      t = t + step_size * f_tt / (f_t - f_tt);
      surface_points_M.push_back((ray_origin_M + ray_dir_M * t).homogeneous());
    }
    if (f_tt < std::abs(0.8f)) {
      // coming closer, reduce step_size
      step_size = step;
    }
    f_t = f_tt;
  }
}

#endif

