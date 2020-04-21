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

#ifndef VTK_IO_H
#define VTK_IO_H
#include <fstream>
#include <sstream>
#include <iostream>
#include "se/utils/math_utils.h"
#include <algorithm>

template <typename T>
void savePointCloud(const T* points_M, const int num_points,
    const char* filename, const Eigen::Matrix4f& T_WM){

  std::stringstream ss_points_W;
  for(int i = 0; i < num_points; ++i){
    Eigen::Vector3f point_W = (T_WM * points_M[i].homogeneous()).head(3);
    ss_points_W << point_W.x() << " "
             << point_W.y() << " "
             << point_W.z() << std::endl;
  }

  std::ofstream f;
  f.open(filename);
  f << "# vtk DataFile Version 1.0" << std::endl;
  f << "vtk mesh generated from KFusion" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET POLYDATA" << std::endl;

  f << "POINTS " << num_points << " FLOAT" << std::endl;
  f << ss_points_W.str();
  f.close();
}

template <typename OctreeT, typename FieldSelector>
void save3DSlice(const OctreeT& in, const Eigen::Vector3i& lower_coord,
    const Eigen::Vector3i& upper_coord, FieldSelector select_value, const int scale, const char* filename){
  std::stringstream ss_x_coord, ss_y_coord, ss_z_coord, ss_scalars;
  std::ofstream f;
  f.open(filename);

  const int stride = 1 << scale;
  const int dimX = std::max(1, (upper_coord.x() - lower_coord.x()) / stride);
  const int dimY = std::max(1, (upper_coord.y() - lower_coord.y()) / stride);
  const int dimZ = std::max(1, (upper_coord.z() - lower_coord.z()) / stride);

  f << "# vtk DataFile Version 1.0" << std::endl;
  f << "vtk mesh generated from KFusion" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET RECTILINEAR_GRID" << std::endl;
  f << "DIMENSIONS " << dimX << " " << dimY << " " << dimZ << std::endl;

  for(int x = lower_coord.x(); x < upper_coord.x(); x += stride)
    ss_x_coord << x << " ";
  for(int y = lower_coord.y(); y < upper_coord.y(); y += stride)
    ss_y_coord << y << " ";
  for(int z = lower_coord.z(); z < upper_coord.z(); z += stride)
    ss_z_coord << z << " ";

  for(int z = lower_coord.z(); z < upper_coord.z(); z += stride)
    for(int y = lower_coord.y(); y < upper_coord.y(); y += stride)
      for(int x = lower_coord.x(); x < upper_coord.x(); x += stride) {
        const auto value = select_value(in.get_fine(x, y, z, scale));
        ss_scalars << value  << std::endl;
      }

  f << "X_COORDINATES " << dimX << " int " << std::endl;
  f << ss_x_coord.str() << std::endl;

  f << "Y_COORDINATES " << dimY << " int " << std::endl;
  f << ss_y_coord.str() << std::endl;

  f << "Z_COORDINATES " << dimZ << " int " << std::endl;
  f << ss_z_coord.str() << std::endl;

  f << "POINT_DATA " << dimX*dimY*dimZ << std::endl;
  f << "SCALARS scalars float 1" << std::endl;
  f << "LOOKUP_TABLE default" << std::endl;
  f << ss_scalars.str() << std::endl;
  f.close();
}
#endif
