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

#ifndef SE_SERIALISE_HPP
#define SE_SERIALISE_HPP
#include <fstream>
#include "../octree_defines.h"
#include "Eigen/Dense"

namespace se {

  template <typename T>
  class Node;

  template <typename T>
  class VoxelBlockFinest;

  template <typename T>
  class VoxelBlockFull;

  template <typename T>
  class VoxelBlockSingle;

  namespace internal {

    /**
     * \brief Write node's data to output file out. We do not serialise child
     * pointers and mask as those will be reconstructed when deserialising.
     *
     * \param[out] out  The binary output file
     * \param[in]  node The node to be serialised
     */
    template <typename T>
    std::ofstream& serialise(std::ofstream& out, Node<T>& node) {
      out.write(reinterpret_cast<char *>(&node.code_), sizeof(key_t));
      out.write(reinterpret_cast<char *>(&node.size_), sizeof(int));
      out.write(reinterpret_cast<char *>(&node.children_mask_), sizeof(unsigned char));
      out.write(reinterpret_cast<char *>(&node.timestamp_), sizeof(unsigned int));
      out.write(reinterpret_cast<char *>(&node.active_), sizeof(bool));
      out.write(reinterpret_cast<char *>(&node.children_data_), sizeof(node.children_data_));
      return out;
    }

    /**
     * \brief Read node's data from input binary file. We do not read child
     * pointers and mask as those will be reconstructed when deserialising.
     *
     * \param[out] node The node to be serialised
     * \param[in]  in   The binary output file
     */
    template <typename T>
    void deserialise(Node<T>& node, std::ifstream& in) {
      in.read(reinterpret_cast<char *>(&node.code_), sizeof(key_t));
      in.read(reinterpret_cast<char *>(&node.size_), sizeof(int));
      in.read(reinterpret_cast<char *>(&node.children_mask_), sizeof(unsigned char));
      in.read(reinterpret_cast<char *>(&node.timestamp_), sizeof(unsigned int));
      in.read(reinterpret_cast<char *>(&node.active_), sizeof(bool));
      in.read(reinterpret_cast<char *>(&node.children_data_), sizeof(node.children_data_));
    }

    /**
     * \brief Write VoxelBlock's data to output file out.
     *
     * \param[out]  out   The binary output file
     * \param[in]   block The node to be serialised
     */
    template <typename T>
    std::ofstream& serialise(std::ofstream& out, VoxelBlockFinest<T>& block) {
      out.write(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      out.write(reinterpret_cast<char *>(&block.size_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      out.write(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      out.write(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      out.write(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      out.write(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      out.write(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.block_data_),
                sizeof(block.block_data_));
      return out;
    }

    /**
     * \brief Read VoxelBlock's data from input file.
     *
     * \param[out] block The block to be deserialised
     * \param[in]  in    The binary input file
     */
    template <typename T>
    void deserialise(VoxelBlockFinest<T>& block, std::ifstream& in) {
      in.read(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      in.read(reinterpret_cast<char *>(&block.size_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      in.read(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      in.read(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      in.read(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      in.read(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      in.read(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.block_data_), sizeof(block.block_data_));
    }

    /**
     * \brief Write VoxelBlock's data to output file out.
     *
     * \param[out]  out   The binary output file
     * \param[in]   block The node to be serialised
     */
    template <typename T>
    std::ofstream& serialise(std::ofstream& out, VoxelBlockFull<T>& block) {
      out.write(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      out.write(reinterpret_cast<char *>(&block.size_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      out.write(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      out.write(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      out.write(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      out.write(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      out.write(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.block_data_),
          sizeof(block.block_data_));
      return out;
    }

    /**
     * \brief Read VoxelBlock's data from input file.
     *
     * \param[out] block The block to be deserialised
     * \param[in]  in    The binary input file
     */
    template <typename T>
    void deserialise(VoxelBlockFull<T>& block, std::ifstream& in) {
      in.read(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      in.read(reinterpret_cast<char *>(&block.size_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      in.read(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      in.read(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      in.read(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      in.read(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      in.read(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.block_data_), sizeof(block.block_data_));
    }

    /**
     * \brief Write VoxelBlock's data to output file out.
     *
     * \param[out]  out   The binary output file
     * \param[in]   block The node to be serialised
     */
    template <typename T>
    std::ofstream& serialise(std::ofstream& out, VoxelBlockSingle<T>& block) {
      out.write(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      out.write(reinterpret_cast<char *>(&block.size_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      out.write(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      out.write(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      out.write(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      out.write(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      out.write(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      out.write(reinterpret_cast<char *>(&block.init_data_), sizeof(typename T::VoxelData));
      if (block.min_scale() != -1) {
        for (int scale = VoxelBlockSingle<T>::max_scale; scale >= block.min_scale(); scale--) {
          int size_at_scale = block.size_li >> scale;
          int num_voxels_at_scale = se::math::cu(size_at_scale);
          out.write(reinterpret_cast<char *>(block.blockData()[VoxelBlockSingle<T>::max_scale - scale]),
              num_voxels_at_scale * sizeof(typename T::VoxelData));
        }
      }
      return out;
    }

    /**
     * \brief Read VoxelBlock's data from input file.
     *
     * \param[out] block The block to be deserialised
     * \param[in]  in    The binary input file
     */
    template <typename T>
    void deserialise(VoxelBlockSingle<T>& block, std::ifstream& in) {
      in.read(reinterpret_cast<char *>(&block.code_), sizeof(key_t));
      in.read(reinterpret_cast<char *>(&block.size_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.children_mask_), sizeof(unsigned char));
      in.read(reinterpret_cast<char *>(&block.timestamp_), sizeof(unsigned int));
      in.read(reinterpret_cast<char *>(&block.active_), sizeof(bool));
      in.read(reinterpret_cast<char *>(&block.children_data_), sizeof(block.children_data_));
      in.read(reinterpret_cast<char *>(&block.coordinates_), sizeof(Eigen::Vector3i));
      in.read(reinterpret_cast<char *>(&block.min_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.current_scale_), sizeof(int));
      in.read(reinterpret_cast<char *>(&block.init_data_), sizeof(typename T::VoxelData));
      if (block.min_scale() != -1) { // Verify that at least some mip-mapped level has been initialised.
        // TODO: Assess if the loaded block is of the same size as the one it's saved to.
        for (int scale = VoxelBlockSingle<T>::max_scale; scale >= block.min_scale(); scale--) {
          int size_at_scale = block.size_li >> scale;
          int num_voxels_at_scale = se::math::cu(size_at_scale);
          typename T::VoxelData* block_data_at_scale = new typename T::VoxelData[num_voxels_at_scale];
          block.blockData().push_back(block_data_at_scale);
          in.read(reinterpret_cast<char *>(block.blockData()[VoxelBlockSingle<T>::max_scale - scale]),
              num_voxels_at_scale * sizeof(typename T::VoxelData));
        }
      }
    }
  }
}
#endif
