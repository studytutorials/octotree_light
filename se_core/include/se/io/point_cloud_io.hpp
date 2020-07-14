// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2016 Emanuele Vespa, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause


#ifndef __POINT_CLOUD_IO_HPP
#define __POINT_CLOUD_IO_HPP

#include <string>
#include <vector>
#include <Eigen/Dense>

#include "se/algorithms/meshing.hpp"



namespace se {

  /**
   * \brief Save the mesh vertices as a PCD file.
   *
   * Documentation for the PCD file format available here
   * https://pcl-tutorials.readthedocs.io/en/latest/pcd_file_format.html.
   *
   * \param[in] mesh     The mesh in map frame to be saved as a point cloud.
   * \param[in] filename The name of the PCD file to create.
   * \param[in] T_WM     The transformation from map to world frame.
   * \return 0 on success, nonzero on error.
   */
  int save_point_cloud_pcd(const std::vector<Triangle>& mesh,
                           const std::string            filename,
                           const Eigen::Matrix4f&       T_WM);

  /**
   * \brief Save the mesh vertices as a PLY file.
   *
   * Documentation for the PLY polygon file format available here
   * https://web.archive.org/web/20161204152348/http://www.dcs.ed.ac.uk/teaching/cs4/www/graphics/Web/ply.html.
   *
   * \param[in] mesh     The mesh in map frame to be saved as a point cloud.
   * \param[in] filename The name of the PLY file to create.
   * \param[in] T_WM     The transformation from map to world frame.
   * \return 0 on success, nonzero on error.
   */
  int save_point_cloud_ply(const std::vector<Triangle>& mesh,
                           const std::string            filename,
                           const Eigen::Matrix4f&       T_WM);

  /**
   * \brief Save the mesh vertices as a VTK file.
   *
   * Documentation for the VTK file format available here
   * https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf.
   *
   * \param[in] mesh     The mesh in map frame to be saved as a point cloud.
   * \param[in] filename The name of the VTK file to create.
   * \param[in] T_WM     The transformation from map to world frame.
   * \return 0 on success, nonzero on error.
   */
  int save_point_cloud_vtk(const std::vector<Triangle>& mesh,
                           const std::string            filename,
                           const Eigen::Matrix4f&       T_WM);

} // namespace se

#include "point_cloud_io_impl.hpp"

#endif // __POINT_CLOUD_IO_HPP
