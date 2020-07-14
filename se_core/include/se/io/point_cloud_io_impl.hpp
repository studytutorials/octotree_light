// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2016 Emanuele Vespa, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __POINT_CLOUD_IO_IMPL_HPP
#define __POINT_CLOUD_IO_IMPL_HPP

#include <iostream>
#include <fstream>
#include <sstream>



int se::save_point_cloud_pcd(const std::vector<Triangle>& mesh,
                             const std::string            filename,
                             const Eigen::Matrix4f&       T_WC) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream ss_points_W;
  int point_count = 0;
  for(size_t i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];
    for(int j = 0; j < 3; ++j) {
      Eigen::Vector3f point_W = (T_WC * triangle_M.vertexes[j].homogeneous()).head(3);
      ss_points_W << point_W.x() << " "
                  << point_W.y() << " "
                  << point_W.z() << std::endl;
      point_count++;
    }
  }

  // Convert from rotation matrix to quaternion.
  const Eigen::Quaternionf q (T_WC.topLeftCorner<3, 3>());

  // Write the PCD header.
  file << "# .PCD v0.7 - Point Cloud Data file format\n";
  file << "VERSION 0.7\n";
  file << "FIELDS x y z\n";
  file << "SIZE 4 4 4\n";
  file << "TYPE F F F\n";
  file << "COUNT 1 1 1\n";
  file << "VIEWPOINT "
       << T_WC(0, 3) << " " << T_WC(1, 3) << " " << T_WC(2, 3) << " "
       << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
  file << "POINTS " << point_count << "\n";
  file << "DATA ascii\n";
  file << ss_points_W.str();

  file.close();
  return 0;
}



int se::save_point_cloud_ply(const std::vector<Triangle>& mesh,
                             const std::string            filename,
                             const Eigen::Matrix4f&       T_WC) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream ss_points_W;
  int point_count = 0;
  for(size_t i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];
    for(int j = 0; j < 3; ++j) {
      Eigen::Vector3f point_W = (T_WC * triangle_M.vertexes[j].homogeneous()).head(3);
      ss_points_W << point_W.x() << " "
                  << point_W.y() << " "
                  << point_W.z() << std::endl;
      point_count++;
    }
  }

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "comment octree structure" << std::endl;
  file << "element vertex " << point_count <<  std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "end_header" << std::endl;
  file << ss_points_W.str();

  file.close();
  return 0;
}



int se::save_point_cloud_vtk(const std::vector<Triangle>& mesh,
                             const std::string            filename,
                             const Eigen::Matrix4f&       T_WC) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream ss_points_W;
  int point_count = 0;
  for(size_t i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];
    for(int j = 0; j < 3; ++j) {
      Eigen::Vector3f point_W = (T_WC * triangle_M.vertexes[j].homogeneous()).head(3);
      ss_points_W << point_W.x() << " "
                  << point_W.y() << " "
                  << point_W.z() << std::endl;
      point_count++;
    }
  }

  file << "# vtk DataFile Version 1.0" << std::endl;
  file << "vtk mesh generated from KFusion" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET POLYDATA" << std::endl;

  file << "POINTS " << point_count << " FLOAT" << std::endl;
  file << ss_points_W.str();

  file.close();
  return 0;
}

#endif // __POINT_CLOUD_IO_IMPL_HPP
