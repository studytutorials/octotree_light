// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2016 Emanuele Vespa, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef MESH_IO_H
#define MESH_IO_H
#include <fstream>
#include <sstream>
#include <iostream>
#include "se/utils/math_utils.h"
#include <algorithm>

namespace se {

  /**
   * \brief Save a mesh as a VTK file.
   *
   * Documentation for the VTK file format available here
   * https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf.
   *
   * \note The resulting mesh is unoptimized and contains many duplicate
   * vertices.
   *
   * \param[in] mesh       The mesh in map frame to be saved.
   * \param[in] filename   The output filename.
   * \param[in] T_WM       The transformation from map to world frame.
   * \param[in] point_data The scalar values of the points/vertices.
   * \param[in] cell_data  The scalar values of the cells/faces.
   * \return 0 on success, nonzero on error.
   */
  static int save_mesh_vtk(const std::vector<Triangle> &mesh,
                           const std::string filename,
                           const Eigen::Matrix4f &T_WM,
                           const float *point_data = nullptr,
                           const float *cell_data = nullptr);

  /**
   * \brief Save a mesh as a PLY file.
   *
   * Documentation for the PLY file format available here
   * http://paulbourke.net/dataformats/ply
   *
   * \note The resulting mesh is unoptimized and contains many duplicate
   * vertices.
   *
   * \param[in] mesh       The mesh in map frame to be saved.
   * \param[in] filename   The output filename.
   * \param[in] T_WM       The transformation from map to world frame.
   * \param[in] point_data The scalar values of the points/vertices.
   * \param[in] cell_data  The scalar values of the cells/faces.
   * \return 0 on success, nonzero on error.
   */
  static int save_mesh_ply(const std::vector<Triangle> &mesh,
                           const std::string filename,
                           const Eigen::Matrix4f &T_WM,
                           const float *point_data = nullptr,
                           const float *cell_data = nullptr);

  /**
   * \brief Save a mesh as a OBJ file.
   *
   * \param[in] mesh     The mesh to be saved.
   * \param[in] filename The output filename.
   * \return 0 on success, nonzero on error.
   */
  static int save_mesh_obj(const std::vector<Triangle> &mesh,
                           const std::string filename);

}

#include "meshing_io_impl.hpp"

#endif
