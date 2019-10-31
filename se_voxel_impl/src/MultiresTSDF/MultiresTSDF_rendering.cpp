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

#include "se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp"

#include <se/utils/math_utils.h>
#include <type_traits>



Eigen::Vector4f MultiresTSDF::raycast(
    const VolumeTemplate<MultiresTSDF, se::Octree>& volume,
    const Eigen::Vector3f&                          origin,
    const Eigen::Vector3f&                          direction,
    const float                                     tnear,
    const float                                     tfar,
    const float                                     mu,
    const float                                     step,
    const float                                     large_step) {

  auto select_depth = [](const auto& val){ return val.x; };
  if (tnear < tfar) {
    // first walk with largesteps until we found a hit
    float t = tnear;
    float step_size = large_step;
    Eigen::Vector3f position = origin + direction * t;
    const int scale = 0;
    auto interp_res = volume.interp(position, scale, select_depth);
    float f_t = interp_res.first;
    float f_tt = 0;
    if (f_t > 0) { // ups, if we were already in it, then don't render anything here
      for (; t < tfar; t += step_size) {
        auto data = volume.get(position, scale);
        if (data.y == 0) {
          step_size = large_step;
          position += step_size * direction;
          continue;
        }
        f_tt = data.x;
        if (f_tt <= 0.1 && f_tt >= -0.5f) {
          interp_res = volume.interp(position, scale, select_depth);
          f_tt = interp_res.first;
        }
        if (f_tt < 0.f)                  // got it, jump out of inner loop
          break;
        step_size = fmaxf(f_tt * mu, step);
        position += step_size * direction;
        f_t = f_tt;
      }
      if (f_tt < 0.f) {
        // got it, calculate accurate intersection
        t = t + step_size * f_tt / (f_t - f_tt);
        Eigen::Vector4f res = (origin + direction * t).homogeneous();
        res.w() = interp_res.second;
        return res;
      }
    }
  }
  return Eigen::Vector4f::Constant(-1.f);
}

