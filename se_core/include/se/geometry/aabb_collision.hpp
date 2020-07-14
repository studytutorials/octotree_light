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
#ifndef AABB_COLLISION_HPP
#define AABB_COLLISION_HPP
#include <cmath>
#include "../utils/math_utils.h"

namespace se {
namespace geometry {
  inline int axis_overlap(int       a_coord,
                          const int a_size,
                          int       b_coord,
                          const int b_size) {
    /* Half plane intersection test */
    a_coord = a_coord + (a_size / 2);
    b_coord = b_coord + (b_size / 2);
    return (std::abs(b_coord - a_coord) > (a_size + b_size) / 2) ? 0 : 1;
  }

  inline int axis_overlap(float       a_coord,
                          const float a_size,
                          float       b_coord,
                          const float b_size) {
    /* Half plane intersection test */
    a_coord = a_coord + (a_size / 2);
    b_coord = b_coord + (b_size / 2);
    return (std::fabs(b_coord - a_coord) > (a_size + b_size) / 2) ? 0 : 1;
  }

  inline int axis_contained(float       a_coord,
                            const float a_size,
                            float       b_coord,
                            const float b_size) {
    /* Segment a includes segment b */
    return (a_coord < b_coord) && ((a_coord + a_size) > (b_coord + b_size));
  }


  inline int aabb_aabb_collision(const Eigen::Vector3i a_coord,
                                 const Eigen::Vector3i a_size,
                                 const Eigen::Vector3i b_coord,
                                 const Eigen::Vector3i b_size){

    return axis_overlap(a_coord.x(), a_size.x(), b_coord.x(), b_size.x()) &&
           axis_overlap(a_coord.y(), a_size.y(), b_coord.y(), b_size.y()) &&
           axis_overlap(a_coord.z(), a_size.z(), b_coord.z(), b_size.z());
  }

  inline int aabb_aabb_inclusion(const Eigen::Vector3i a_coord,
                                 const Eigen::Vector3i a_size,
                                 const Eigen::Vector3i b_coord,
                                 const Eigen::Vector3i b_size){
    /* Box a contains box b */
    return axis_contained(a_coord.x(), a_size.x(), b_coord.x(), b_size.x()) &&
           axis_contained(a_coord.y(), a_size.y(), b_coord.y(), b_size.y()) &&
           axis_contained(a_coord.z(), a_size.z(), b_coord.z(), b_size.z());
  }
}
}
#endif
