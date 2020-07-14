// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include "se/point_cloud_utils.hpp"

#include <fstream>
#include <iostream>



int se::save_point_cloud_pcd(se::Image<Eigen::Vector3f>& point_cloud,
                             const std::string&          filename,
                             const Eigen::Matrix4f&      T_WC) {
  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
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
  file << "WIDTH " << point_cloud.width() << "\n";
  file << "HEIGHT " << point_cloud.height() << "\n";
  file << "VIEWPOINT "
      << T_WC(0, 3) << " " << T_WC(1, 3) << " " << T_WC(2, 3) << " "
      << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << "\n";
  file << "POINTS " << point_cloud.size() << "\n";
  file << "DATA ascii\n";

  // Write the point data.
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    file << point_cloud[i].x() << " "
         << point_cloud[i].y() << " "
         << point_cloud[i].z() << "\n";
  }

  file.close();
  return 0;
}



int se::save_point_cloud_ply(se::Image<Eigen::Vector3f>& point_cloud,
                             const std::string&          filename,
                             const Eigen::Matrix4f&      T_WC) {
  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  file << "ply" << std::endl;
  file << "format ascii 1.0" << std::endl;
  file << "comment octree structure" << std::endl;
  file << "element point " << point_cloud.size() <<  std::endl;
  file << "property float x" << std::endl;
  file << "property float y" << std::endl;
  file << "property float z" << std::endl;
  file << "end_header" << std::endl;

  // Write the point data.
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const Eigen::Vector3f point_W = (T_WC * point_cloud[i].homogeneous()).head(3);
    file << point_W.x() << " "
         << point_W.y() << " "
         << point_W.z() << "\n";
  }

  file.close();
  return 0;
}



int se::save_point_cloud_vtk(se::Image<Eigen::Vector3f>& point_cloud,
                             const std::string&          filename,
                             const Eigen::Matrix4f&      T_WC){

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  file << "# vtk DataFile Version 1.0" << std::endl;
  file << "vtk mesh generated from KFusion" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET POLYDATA" << std::endl;

  file << "POINTS " << point_cloud.size() << " FLOAT" << std::endl;

  // Write the point data.
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const Eigen::Vector3f point_W = (T_WC * point_cloud[i].homogeneous()).head(3);
    file << point_W.x() << " "
         << point_W.y() << " "
         << point_W.z() << "\n";
  }

  file.close();
  return 0;
}
