// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __SENSOR_HPP
#define __SENSOR_HPP

#include <cmath>

#include <Eigen/Dense>
#include <srl/projection/NoDistortion.hpp>
#include <srl/projection/OusterLidar.hpp>
#include <srl/projection/PinholeCamera.hpp>



namespace se {

  struct SensorConfig {
    // General
    int width = 0;
    int height = 0;
    bool left_hand_frame = false;
    float near_plane = 0.f;
    float far_plane = INFINITY;
    float mu = 0.1f;
    // Pinhole camera
    float fx = nan("");
    float fy = nan("");
    float cx = nan("");
    float cy = nan("");
    // LIDAR
    Eigen::VectorXf beam_azimuth_angles = Eigen::VectorXf(0);
    Eigen::VectorXf beam_elevation_angles = Eigen::VectorXf(0);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  struct PinholeCamera {
    PinholeCamera(const SensorConfig& c);

    PinholeCamera(const PinholeCamera& pinhole_camera,
                  const float          scaling_factor);

    /**
     * \brief Computes the scale corresponding to the back-projected pixel size
     * in voxel space
     * \param[in] dist            Distance from the camera to the voxel
     *                            block centre.
     * \param[in] voxel_dim       The voxel edge length in meters.
     * \param[in] last_scale      Scale from which propagate up voxel
     *                            values.
     * \param[in] min_scale
     * \param[in] max_block_scale The maximum allowed scale within a
     *                            VoxelBlock.
     * \return The scale that should be used for the integration.
     */
    int computeIntegrationScale(const float dist,
                                const float voxel_dim,
                                const int   last_scale,
                                const int   min_scale,
                                const int   max_block_scale) const;



    srl::projection::PinholeCamera<srl::projection::NoDistortion> model;
    bool  left_hand_frame;
    float near_plane;
    float far_plane;
    float mu;
    float scaled_pixel;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };



  struct OusterLidar {
    OusterLidar(const SensorConfig& c);

    OusterLidar(const OusterLidar& ouster_lidar,
                const float        scaling_factor);

    int computeIntegrationScale(const float,
                                const float,
                                const int,
                                const int,
                                const int) const {return 0;};



    srl::projection::OusterLidar model;
    bool  left_hand_frame;
    float near_plane;
    float far_plane;
    float mu;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

} // namespace se

#endif

