// SPDX-FileCopyrightText: 2018-2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2016 Emanuele Vespa, ETH Zürich
// SPDX-FileCopyrightText: 2019-2020 Nils Funk, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __MESHING_IO_IMPL_HPP
#define __MESHING_IO_IMPL_HPP

int se::save_mesh_vtk(const std::vector<Triangle>& mesh,
                      const std::string            filename,
                      const Eigen::Matrix4f&       T_WM,
                      const float*                 point_data,
                      const float*                 cell_data) {

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream ss_points_W;
  std::stringstream ss_polygons;
  std::stringstream ss_point_data;
  std::stringstream ss_cell_data;
  int point_count = 0;
  int triangle_count = 0;
  bool has_point_data = point_data != nullptr;
  bool has_cell_data = cell_data != nullptr;

  for(unsigned int i = 0; i < mesh.size(); ++i ){
    const Triangle& triangle_M = mesh[i];

    Eigen::Vector3f vertex_0_W = (T_WM * triangle_M.vertexes[0].homogeneous()).head(3);
    Eigen::Vector3f vertex_1_W = (T_WM * triangle_M.vertexes[1].homogeneous()).head(3);
    Eigen::Vector3f vertex_2_W = (T_WM * triangle_M.vertexes[2].homogeneous()).head(3);

    ss_points_W << vertex_0_W.x() << " "
                << vertex_0_W.y() << " "
                << vertex_0_W.z() << std::endl;

    ss_points_W << vertex_1_W.x() << " "
                << vertex_1_W.y() << " "
                << vertex_1_W.z() << std::endl;

    ss_points_W << vertex_2_W.x() << " "
                << vertex_2_W.y() << " "
                << vertex_2_W.z() << std::endl;

    ss_polygons << "3 " << point_count << " " << point_count+1 <<
                " " << point_count+2 << std::endl;

    if(has_point_data){
      ss_point_data << point_data[i*3] << std::endl;
      ss_point_data << point_data[i*3 + 1] << std::endl;
      ss_point_data << point_data[i*3 + 2] << std::endl;
    }

    if(has_cell_data){
      ss_cell_data << cell_data[i] << std::endl;
    }

    point_count +=3;
    triangle_count++;
  }

  file << "# vtk DataFile Version 1.0" << std::endl;
  file << "vtk mesh generated from KFusion" << std::endl;
  file << "ASCII" << std::endl;
  file << "DATASET POLYDATA" << std::endl;

  file << "POINTS " << point_count << " FLOAT" << std::endl;
  file << ss_points_W.str();

  file << "POLYGONS " << triangle_count << " " << triangle_count * 4 << std::endl;
  file << ss_polygons.str() << std::endl;
  if(has_point_data){
    file << "POINT_DATA " << point_count << std::endl;
    file << "SCALARS vertex_scalars float 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    file << ss_point_data.str();
  }

  if(has_cell_data){
    file << "CELL_DATA " << triangle_count << std::endl;
    file << "SCALARS cell_scalars float 1" << std::endl;
    file << "LOOKUP_TABLE default" << std::endl;
    file << ss_cell_data.str();
  }

  file.close();
  return 0;
}



int se::save_mesh_obj(const std::vector<Triangle>& mesh,
                      const std::string            filename){

  // Open the file for writing.
  std::ofstream file (filename.c_str());
  if (!file.is_open()) {
    std::cerr << "Unable to write file " << filename << "\n";
    return 1;
  }

  std::stringstream points_M;
  std::stringstream faces;
  int point_count = 0;
  int face_count = 0;

  for(unsigned int i = 0; i < mesh.size(); i++){
    const Triangle& triangle_M = mesh[i];
    points_M << "v " << triangle_M.vertexes[0].x() << " "
             << triangle_M.vertexes[0].y() << " "
             << triangle_M.vertexes[0].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[1].x() << " "
             << triangle_M.vertexes[1].y() << " "
             << triangle_M.vertexes[1].z() << std::endl;
    points_M << "v " << triangle_M.vertexes[2].x() << " "
             << triangle_M.vertexes[2].y() << " "
             << triangle_M.vertexes[2].z() << std::endl;

    faces  << "f " << (face_count*3)+1 << " " << (face_count*3)+2
           << " " << (face_count*3)+3 << std::endl;

    point_count +=3;
    face_count += 1;
  }

  file << "# OBJ file format with ext .obj" << std::endl;
  file << "# vertex count = " << point_count << std::endl;
  file << "# face count = " << face_count << std::endl;
  file << points_M.str();
  file << faces.str();

  file.close();
  return 0;
}

#endif // __MESHING_IO_IMPL_HPP
