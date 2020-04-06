// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/sensor.hpp"

#include <cassert>

// Explicit template class instantiation
template class srl::projection::PinholeCamera<srl::projection::NoDistortion>;

// Used for initializing a PinholeCamera.
const srl::projection::NoDistortion _distortion;



se::PinholeCamera::PinholeCamera(const SensorConfig& c)
    : sensor(c.width, c.height, c.fx, c.fy, c.cx, c.cy, _distortion),
      near_plane(c.near_plane), far_plane(c.far_plane), mu(c.mu) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(c.mu > 0.f);
  assert(!isnan(c.fx));
  assert(!isnan(c.fy));
  assert(!isnan(c.cx));
  assert(!isnan(c.cy));
}



se::OusterLidar::OusterLidar(const SensorConfig& c)
    : sensor(c.width, c.height, c.beam_azimuth_angles, c.beam_elevation_angles),
      near_plane(c.near_plane), far_plane(c.far_plane), mu(c.mu) {
  assert(c.width  > 0);
  assert(c.height > 0);
  assert(c.near_plane >= 0.f);
  assert(c.far_plane > c.near_plane);
  assert(c.mu > 0.f);
  assert(c.beam_azimuth_angles.size()   > 0);
  assert(c.beam_elevation_angles.size() > 0);
}

