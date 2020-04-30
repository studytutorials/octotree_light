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

#ifndef MESHING_HPP
#define MESHING_HPP
#include "../octree.hpp"
#include "edge_tables.h"

namespace se {
namespace meshing {
  enum status : uint8_t {
    OUTSIDE = 0x0,
    UNKNOWN = 0xFE, // 254
    INSIDE = 0xFF, // 255
  };

  void savePointCloudPly(const std::vector<Triangle>& mesh,
      const char* filename, const Eigen::Matrix4f& T_WM) {
    std::stringstream ss_points_W;
    int point_count = 0;
    for(size_t i = 0; i < mesh.size(); ++i ){
      const Triangle& triangle_M = mesh[i];
      for(int j = 0; j < 3; ++j) {
        Eigen::Vector3f point_W = (T_WM * triangle_M.vertexes[j].homogeneous()).head(3);
        ss_points_W << point_W.x() << " "
                    << point_W.y() << " "
                    << point_W.z() << std::endl;
        point_count++;
      }
    }

    std::ofstream f;
    f.open(std::string(filename).c_str());

    f << "ply" << std::endl;
    f << "format ascii 1.0" << std::endl;
    f << "comment octree structure" << std::endl;
    f << "element vertex " << point_count <<  std::endl;
    f << "property float x" << std::endl;
    f << "property float y" << std::endl;
    f << "property float z" << std::endl;
    f << "end_header" << std::endl;
    f << ss_points_W.str();
  }

  template <typename OctreeT, typename FieldSelector>
    inline Eigen::Vector3f compute_intersection(const OctreeT& map, FieldSelector select,
        const Eigen::Vector3i& source, const Eigen::Vector3i& dest){
      const float voxel_dim = map.dim() / map.size();
      Eigen::Vector3f s = Eigen::Vector3f(source.x() * voxel_dim, source.y() * voxel_dim, source.z() * voxel_dim);
      Eigen::Vector3f d = Eigen::Vector3f(dest.x() * voxel_dim, dest.y() * voxel_dim, dest.z() * voxel_dim);
      float value_0 = select(map.get_fine(source.x(), source.y(), source.z()));
      float value_1 = select(map.get_fine(dest.x(), dest.y(), dest.z()));
      return s + (0.0 - value_0) * (d - s) / (value_1 - value_0);
    }

  template <typename OctreeT, typename FieldSelector>
    inline Eigen::Vector3f interp_vertexes(const OctreeT& map, FieldSelector select_value,
        const unsigned x, const unsigned y, const unsigned z, const int edge){
      switch(edge){
        case 0:  return compute_intersection(map, select_value, Eigen::Vector3i(x, y, z),
                     Eigen::Vector3i(x + 1, y, z));
        case 1:  return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y, z),
                     Eigen::Vector3i(x + 1, y, z + 1));
        case 2:  return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y, z + 1),
                     Eigen::Vector3i(x, y, z + 1));
        case 3:  return compute_intersection(map, select_value, Eigen::Vector3i(x, y, z),
                     Eigen::Vector3i(x, y, z + 1));
        case 4:  return compute_intersection(map, select_value, Eigen::Vector3i(x, y + 1, z),
                     Eigen::Vector3i(x + 1, y + 1, z));
        case 5:  return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y + 1, z),
                     Eigen::Vector3i(x + 1, y + 1, z + 1));
        case 6:  return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y + 1, z + 1),
                     Eigen::Vector3i(x, y + 1, z + 1));
        case 7:  return compute_intersection(map, select_value, Eigen::Vector3i(x, y + 1, z),
                     Eigen::Vector3i(x, y + 1, z + 1));

        case 8:  return compute_intersection(map, select_value, Eigen::Vector3i(x, y, z),
                     Eigen::Vector3i(x, y + 1, z));
        case 9:  return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y, z),
                     Eigen::Vector3i(x + 1, y + 1, z));
        case 10: return compute_intersection(map, select_value, Eigen::Vector3i(x + 1, y, z + 1),
                     Eigen::Vector3i(x + 1, y + 1, z + 1));
        case 11: return compute_intersection(map, select_value, Eigen::Vector3i(x, y, z + 1),
                     Eigen::Vector3i(x, y + 1, z + 1));
      }
      return Eigen::Vector3f::Constant(0);
    }

  template <typename FieldType, typename DataT>
    inline void gather_data( const se::VoxelBlock<FieldType>* cached, DataT data[8],
        const int x, const int y, const int z) {
      data[0] = cached->data(Eigen::Vector3i(x, y, z));
      data[1] = cached->data(Eigen::Vector3i(x + 1, y, z));
      data[2] = cached->data(Eigen::Vector3i(x + 1, y, z + 1));
      data[3] = cached->data(Eigen::Vector3i(x, y, z + 1));
      data[4] = cached->data(Eigen::Vector3i(x, y + 1, z));
      data[5] = cached->data(Eigen::Vector3i(x + 1, y + 1, z));
      data[6] = cached->data(Eigen::Vector3i(x + 1, y + 1, z + 1));
      data[7] = cached->data(Eigen::Vector3i(x, y + 1, z + 1));
    }

  template <typename FieldType, template <typename FieldT> class OctreeT, typename PointT>
  inline void gather_data(const OctreeT<FieldType>& map, PointT data[8],
                 const int x, const int y, const int z) {
               data[0] = map.get_fine(x, y, z);
               data[1] = map.get_fine(x + 1, y, z);
               data[2] = map.get_fine(x + 1, y, z + 1);
               data[3] = map.get_fine(x, y, z + 1);
               data[4] = map.get_fine(x, y + 1, z);
               data[5] = map.get_fine(x + 1, y + 1, z);
               data[6] = map.get_fine(x + 1, y + 1, z + 1);
               data[7] = map.get_fine(x, y + 1, z + 1);
             }

  template <typename FieldType, template <typename FieldT> class OctreeT,
  typename InsidePredicate>
  uint8_t compute_index(const OctreeT<FieldType>& map,
  const se::VoxelBlock<FieldType>* cached, InsidePredicate inside,
  const unsigned x, const unsigned y, const unsigned z){
    unsigned int block_size =  se::VoxelBlock<FieldType>::size;
    unsigned int local = ((x % block_size == block_size - 1) << 2) |
      ((y % block_size == block_size - 1) << 1) |
      ((z % block_size) == block_size - 1);

    typename FieldType::VoxelData data[8];
    if(!local) gather_data(cached, data, x, y, z);
    else gather_data(map, data, x, y, z);

    uint8_t index = 0;

    if(data[0].y == 0.f) return 0;
    if(data[1].y == 0.f) return 0;
    if(data[2].y == 0.f) return 0;
    if(data[3].y == 0.f) return 0;
    if(data[4].y == 0.f) return 0;
    if(data[5].y == 0.f) return 0;
    if(data[6].y == 0.f) return 0;
    if(data[7].y == 0.f) return 0;

    if(inside(data[0])) index |= 1;
    if(inside(data[1])) index |= 2;
    if(inside(data[2])) index |= 4;
    if(inside(data[3])) index |= 8;
    if(inside(data[4])) index |= 16;
    if(inside(data[5])) index |= 32;
    if(inside(data[6])) index |= 64;
    if(inside(data[7])) index |= 128;
    // std::cerr << std::endl << std::endl;

    return index;
  }

  inline bool checkVertex(const Eigen::Vector3f& vertex_M, const float dim){
    return (vertex_M.x() <= 0 || vertex_M.y() <=0 || vertex_M.z() <= 0 ||
            vertex_M.x() > dim || vertex_M.y() > dim || vertex_M.z() > dim);
  }

}
namespace algorithms {
  template <typename FieldType, typename FieldSelector,
            typename InsidePredicate, typename TriangleType>
    void marching_cube(Octree<FieldType>& map, FieldSelector select_value,
        InsidePredicate inside, std::vector<TriangleType>& triangles)
    {

      using namespace meshing;
      std::vector<se::VoxelBlock<FieldType>*> block_list;
      std::mutex lck;
      const int map_size = map.size();
      const float map_dim = map.dim();
      map.getBlockList(block_list, false);
      std::cout << "Blocklist size: " << block_list.size() << std::endl;


#pragma omp parallel for
      for (size_t i = 0; i < block_list.size(); i++) {
        se::VoxelBlock<FieldType>* block = static_cast<se::VoxelBlock<FieldType> *>(block_list[i]);
        const int block_size = se::VoxelBlock<FieldType>::size;
        const Eigen::Vector3i& start_coord = block->coordinates();
        const Eigen::Vector3i last_coord =
          (block->coordinates() + Eigen::Vector3i::Constant(block_size)).cwiseMin(
              Eigen::Vector3i::Constant(map_size-1));
        for (int x = start_coord.x(); x < last_coord.x(); x++) {
          for (int y = start_coord.y(); y < last_coord.y(); y++) {
            for (int z = start_coord.z(); z < last_coord.z(); z++) {

              const uint8_t idx = meshing::compute_index(map, block, inside, x, y, z);

              int* edges = triTable[idx];
              for (unsigned int e = 0; edges[e] != -1 && e < 16; e += 3) {
                Eigen::Vector3f vertex_0 = interp_vertexes(map, select_value, x, y, z, edges[e]);
                Eigen::Vector3f vertex_1 = interp_vertexes(map, select_value, x, y, z, edges[e + 1]);
                Eigen::Vector3f vertex_2 = interp_vertexes(map, select_value, x, y, z, edges[e + 2]);
                if (checkVertex(vertex_0, map_dim) || checkVertex(vertex_1, map_dim) || checkVertex(vertex_2, map_dim))
                  continue;
                Triangle temp = Triangle();
                temp.vertexes[0] = vertex_0;
                temp.vertexes[1] = vertex_1;
                temp.vertexes[2] = vertex_2;
                lck.lock();
                triangles.push_back(temp);
                lck.unlock();
              }
            }
          }
        }
      }
    }
}
}
#endif
