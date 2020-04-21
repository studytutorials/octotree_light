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

#ifndef PLY_IO_HPP
#define PLY_IO_HPP
#include <fstream>
#include <sstream>
#include "se/octree.hpp"
#include "se/node_iterator.hpp"
#include <Eigen/Dense>

/*! \file */
namespace se {
  template <typename T>
  /*! \brief Write the octree structure to file in ply format.
   * Aggregated voxel blocks are written as a single octant of size block_size^3.
   * \param filename output filename
   * \param octree octree map
   */
    void print_octree(const char* filename, const se::Octree<T>& octree) {
      std::stringstream ss_nodes_corners;
      std::stringstream ss_faces;
      se::node_iterator<T> it(octree);
      se::Node<T>* node = it.next();
      const int voxel_depth = octree.voxelDepth();
      int nodes_corners_count = 0;
      int faces_count  = 0;
      while(node) {
        const Eigen::Vector3i node_coord = se::keyops::decode(node->code_);
        const int node_size = 1 << (voxel_depth - se::keyops::depth(node->code_));

        Eigen::Vector3f node_corners[8];
        node_corners[0] =  node_coord.cast<float>();
        node_corners[1] = (node_coord + Eigen::Vector3i(node_size, 0, 0)).cast<float>();
        node_corners[2] = (node_coord + Eigen::Vector3i(0, node_size, 0)).cast<float>();
        node_corners[3] = (node_coord + Eigen::Vector3i(node_size, node_size, 0)).cast<float>();
        node_corners[4] = (node_coord + Eigen::Vector3i(0, 0, node_size)).cast<float>();
        node_corners[5] = (node_coord + Eigen::Vector3i(node_size, 0, node_size)).cast<float>();
        node_corners[6] = (node_coord + Eigen::Vector3i(0, node_size, node_size)).cast<float>();
        node_corners[7] = (node_coord + Eigen::Vector3i(node_size, node_size, node_size)).cast<float>();

        for(int i = 0; i < 8; ++i) {
          ss_nodes_corners << node_corners[i].x() << " "
                           << node_corners[i].y() << " "
                           << node_corners[i].z() << std::endl;
        }

        ss_faces << "4 " << nodes_corners_count     << " " << nodes_corners_count + 1
                 << " "  << nodes_corners_count + 3 << " " << nodes_corners_count + 2 << std::endl;

        ss_faces << "4 " << nodes_corners_count + 1 << " " << nodes_corners_count + 5
                 << " "  << nodes_corners_count + 7 << " " << nodes_corners_count + 3 << std::endl;

        ss_faces << "4 " << nodes_corners_count + 5 << " " << nodes_corners_count + 7
                 << " "  << nodes_corners_count + 6 << " " << nodes_corners_count + 4 << std::endl;

        ss_faces << "4 " << nodes_corners_count     << " " << nodes_corners_count + 2
                 << " "  << nodes_corners_count + 6 << " " << nodes_corners_count + 4 << std::endl;

        ss_faces << "4 " << nodes_corners_count     << " " << nodes_corners_count + 1
                 << " "  << nodes_corners_count + 5 << " " << nodes_corners_count + 4 << std::endl;

        ss_faces << "4 " << nodes_corners_count + 2 << " " << nodes_corners_count + 3
                 << " "  << nodes_corners_count + 7 << " " << nodes_corners_count + 6 << std::endl;

        nodes_corners_count += 8;
        faces_count  += 6;
        node = it.next();
      }

      {
        std::ofstream f;
        f.open(std::string(filename).c_str());

        f << "ply" << std::endl;
        f << "format ascii 1.0" << std::endl;
        f << "comment octree structure" << std::endl;
        f << "element point " << nodes_corners_count <<  std::endl;
        f << "property float x" << std::endl;
        f << "property float y" << std::endl;
        f << "property float z" << std::endl;
        f << "element face " << faces_count << std::endl;
        f << "property list uchar int point_index" << std::endl;
        f << "end_header" << std::endl;
        f << ss_nodes_corners.str();
        f << ss_faces.str();
      }
    }

  template <typename T>
    void savePointCloudPly(const T* points_M, const int num_points,
        const char* filename, const Eigen::Matrix4f& T_WM) {
      std::stringstream ss_points_W;
      for(int i = 0; i < num_points; ++i){
        Eigen::Vector3f point_W = (T_WM * points_M[i].homogeneous()).head(3);
        ss_points_W << point_W.x() << " "
                 << point_W.y() << " "
                 << point_W.z() << std::endl;
      }

      std::ofstream f;
      f.open(std::string(filename).c_str());

      f << "ply" << std::endl;
      f << "format ascii 1.0" << std::endl;
      f << "comment octree structure" << std::endl;
      f << "element point " << num_points <<  std::endl;
      f << "property float x" << std::endl;
      f << "property float y" << std::endl;
      f << "property float z" << std::endl;
      f << "end_header" << std::endl;
      f << ss_points_W.str();
    }
}
#endif
