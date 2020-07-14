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

#ifndef __TRACKING_HPP
#define __TRACKING_HPP

#include "se/utils/math_utils.h"
#include "se/timings.h"
#include "se/perfstats.h"
#include "se/commons.h"
#include "se/image/image.hpp"
#include "se/image_utils.hpp"
#include "se/sensor_implementation.hpp"



struct TrackData {
	int result;
	float error;
	float J[6];

    TrackData()
      : result(0),
      error(0.0f),
      J{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f} {}
};



void new_reduce(const int              block_index,
                float*                 output_data,
                const Eigen::Vector2i& output_res,
                TrackData*             J_data,
                const Eigen::Vector2i& J_res);



void reduceKernel(float*                 output_data,
                  const Eigen::Vector2i& output_res,
                  TrackData*             J_data,
                  const Eigen::Vector2i& J_res);


/**
 * \brief Compute the tracking data to update the current camera pose via ICP
 *
 * param[out] output_data               The array containing the tracking data.
 * param[in]  input_point_cloud_C       The point cloud input provided by the sensor.
 * param[in]  input_normals_C           The normals of the point cloud input provided by the sensor.
 * param[in]  ref_surface_point_cloud_M The surface point cloud in map frame computed from the previous raycasting pose.
 * param[in]  ref_surface_normals_M     The surface normals in map frame computed from the previous raycasting pose.
 * param[in]  T_MC                      The current approximation of the camera pose.
 * param[in]  T_MC_ref                  The raycasting pose the surface point cloud and normal was computed from.
 * param[in]  sensor                    The sensor model used.
 * param[in]  dist_threshold            The maximum abs distance between the input point and the projected surface point.
 * param[in]  normal_threshold          The maximum dot product between the input normal and the projected normal.
 */
void trackKernel(TrackData*                        output_data,
                 const se::Image<Eigen::Vector3f>& input_point_cloud_C,
                 const se::Image<Eigen::Vector3f>& input_normals_C,
                 const se::Image<Eigen::Vector3f>& surface_point_cloud_M,
                 const se::Image<Eigen::Vector3f>& surface_normals_M,
                 const Eigen::Matrix4f&            T_MC,
                 const Eigen::Matrix4f&            T_MC_ref,
                 const SensorImpl&                 sensor,
                 const float                       dist_threshold,
                 const float                       normal_threshold);



bool updatePoseKernel(Eigen::Matrix4f& T_MC,
                      const float*     reduction_output,
                      const float      icp_threshold);



bool checkPoseKernel(Eigen::Matrix4f&       T_MC,
                     Eigen::Matrix4f&       previous_T_MC,
                     const float*           reduction_output_data,
                     const Eigen::Vector2i& reduction_output_res,
                     const float            track_threshold);

#endif

