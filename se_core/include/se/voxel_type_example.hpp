/*
 * Copyright 2019 Sotiris Papatheodorou, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __VOXEL_TYPE_EXAMPLE_HPP
#define __VOXEL_TYPE_EXAMPLE_HPP



/**
 * An example voxel type used as the template parameter for se::Octree. This is
 * a minimal working example.
 *
 * \note There is usually no reason to create an instance of this struct, it is
 * only meant to be passed as a template parameter to se::Octree and related
 * functions/classes.
 */
struct ExampleVoxelT {
  /**
   * The declaration of the struct stored in each octree voxel. It may contain
   * additional members if desired.
   *
   * \warning The struct name must always be `VoxelData`.
   */
  struct VoxelData {
    float x;
  };



  /**
   * Returns the value stored in newly created voxels.
   *
   * \warning This function declaration is required and the function name must
   * always be `initValue`.
   */
  static inline VoxelData initData() { return 1.f; }



  /**
   * Returns a value corresponding to invalid voxels.
   *
   * \warning This function declaration is required and the function name must
   * always be `empty`.
   */
  static inline VoxelData invalid()     { return 0.f; }
};

#endif

