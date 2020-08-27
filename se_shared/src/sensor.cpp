// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/sensor.hpp"

#include <cassert>

// Explicit template class instantiation
template class srl::projection::PinholeCamera<srl::projection::NoDistortion>;

// Used for initializing a PinholeCamera.
const srl::projection::NoDistortion _distortion;



se::PinholeCamera::PinholeCamera(const SensorConfig& c)
    : model(c.width, c.height, c.fx, c.fy, c.cx, c.cy, _distortion),
      left_hand_frame(c.left_hand_frame), near_plane(c.near_plane), far_plane(c.far_plane), scaled_pixel(1 / c.fx) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(!std::isnan(c.fx));
  assert(!std::isnan(c.fy));
  assert(!std::isnan(c.cx));
  assert(!std::isnan(c.cy));
}

se::PinholeCamera::PinholeCamera(const PinholeCamera& pc, const float sf)
    : model(pc.model.imageWidth() * sf, pc.model.imageHeight() * sf,
            pc.model.focalLengthU() * sf, pc.model.focalLengthV() * sf,
            pc.model.imageCenterU() * sf, pc.model.imageCenterV() * sf, _distortion),
            left_hand_frame(pc.left_hand_frame), near_plane(pc.near_plane), far_plane(pc.far_plane) {
}

int se::PinholeCamera::computeIntegrationScale(
    const Eigen::Vector3f& block_centre,
    const float            voxel_dim,
    const int              last_scale,
    const int              min_scale,
    const int              max_block_scale) const {
  const float dist = block_centre.z();
  // Compute the side length in metres of a pixel projected dist metres from the camera
  const float pixel_dim = dist * scaled_pixel;
  const float pv_ratio = pixel_dim / voxel_dim;
  int scale = 0;
  if (pv_ratio < 1.5) {
    scale = 0;
  } else if (pv_ratio < 3) {
    scale = 1;
  } else if (pv_ratio < 6) {
    scale = 2;
  } else {
    scale = 3;
  }
  scale = std::min(scale, max_block_scale);

  Eigen::Vector3f block_centre_hyst = block_centre;
  bool recompute = false;
  if (scale > last_scale && min_scale != -1) {
    block_centre_hyst.z() -= 0.25;
    recompute = true;
  } else if (scale < last_scale && min_scale != -1) {
    block_centre_hyst.z() += 0.25;
    recompute = true;
  }

  if (recompute) {
    return computeIntegrationScale(block_centre_hyst, voxel_dim, last_scale, -1, max_block_scale);
  } else {
    return scale;
  }
}

float se::PinholeCamera::nearDist(const Eigen::Vector3f& ray_C) const {
  return near_plane / ray_C.normalized().z();
}

float se::PinholeCamera::farDist(const Eigen::Vector3f& ray_C) const {
  return far_plane / ray_C.normalized().z();
}

float se::PinholeCamera::measurementFromPoint(const Eigen::Vector3f& point_C) const {
  return point_C.z();
}



se::OusterLidar::OusterLidar(const SensorConfig& c)
    : model(c.width, c.height, c.beam_azimuth_angles, c.beam_elevation_angles),
      left_hand_frame(c.left_hand_frame), near_plane(c.near_plane), far_plane(c.far_plane) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(c.beam_azimuth_angles.size()   > 0);
  assert(c.beam_elevation_angles.size() > 0);
  float min_elevation_angle = fabsf(c.beam_elevation_angles[1] - c.beam_elevation_angles[0]);
  for (int i = 2; i < c.beam_elevation_angles.size(); i++) {
    const float diff = fabsf(c.beam_elevation_angles[i-1] - c.beam_elevation_angles[i]);
    if (diff < min_elevation_angle) {
      min_elevation_angle = diff;
    }
  }
  const float azimuth_angle = 360.0f / c.width;
  min_ray_angle = std::min(min_elevation_angle, azimuth_angle);
}

se::OusterLidar::OusterLidar(const OusterLidar& ol, const float sf)
    : model(ol.model.imageWidth() * sf, ol.model.imageHeight() * sf,
            ol.model.beamAzimuthAngles(), ol.model.beamElevationAngles()), // TODO: Does the beam need to be scaled too?
            left_hand_frame(ol.left_hand_frame), near_plane(ol.near_plane), far_plane(ol.far_plane) {
}

int se::OusterLidar::computeIntegrationScale(
    const Eigen::Vector3f& block_centre,
    const float voxel_dim,
    const int   last_scale,
    const int   min_scale,
    const int   max_block_scale) const {
  constexpr float deg_to_rad = M_PI / 180.0f;
  const float dist = block_centre.norm();
  // Compute the side length in metres of a pixel projected dist metres from
  // the camera. This computes the chord length corresponding to the ray angle
  // at distance dist.
  const float pixel_dim = 2.0f * dist * std::tan(min_ray_angle / 2.0f * deg_to_rad);
  // Compute the ratio using the worst case voxel_dim (space diagonal)
  const float pv_ratio = pixel_dim / (std::sqrt(3) * voxel_dim);
  int scale = 0;
  if (pv_ratio < 1.0f) {
    scale = 0;
  } else if (pv_ratio < 2.0f) {
    scale = 1;
  } else if (pv_ratio < 4.0f) {
    scale = 2;
  } else {
    scale = 3;
  }
  scale = std::min(scale, max_block_scale);

  Eigen::Vector3f block_centre_hyst = block_centre;
  bool recompute = false;
  if (scale > last_scale && min_scale != -1) {
    block_centre_hyst -= 0.25 * block_centre_hyst.normalized();
    recompute = true;
  } else if (scale < last_scale && min_scale != -1) {
    block_centre_hyst += 0.25 * block_centre_hyst.normalized();
    recompute = true;
  }

  if (recompute) {
    return computeIntegrationScale(block_centre_hyst, voxel_dim, last_scale, -1, max_block_scale);
  } else {
    return scale;
  }
}

float se::OusterLidar::nearDist(const Eigen::Vector3f&) const {
  return near_plane;
}

float se::OusterLidar::farDist(const Eigen::Vector3f&) const {
  return far_plane;
}

float se::OusterLidar::measurementFromPoint(const Eigen::Vector3f& point_C) const {
  return point_C.norm();
}

