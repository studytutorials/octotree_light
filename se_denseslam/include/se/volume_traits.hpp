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
#ifndef VOLUME_H
#define VOLUME_H



/******************************************************************************
 *
 * KFusion Truncated Signed Distance Function voxel traits
 *
****************************************************************************/

struct SDF {
  typedef struct {
    float x;
    float y;
  } VoxelData;

  static inline VoxelData empty()     { return {1.f, -1.f}; }
  static inline VoxelData initValue() { return {1.f,  0.f}; }
};

/******************************************************************************
 *
 * Bayesian Fusion voxel traits and algorithm specificic defines
 *
****************************************************************************/

struct OFusion {
  typedef struct {
    float  x;
    double y;
  } VoxelData;

  static inline VoxelData empty()     { return {0.f, 0.f}; }
  static inline VoxelData initValue() { return {0.f, 0.f}; }
};

// Windowing parameters
#define DELTA_T   1.f
#define CAPITAL_T 4.f

#define INTERP_THRESH 0.05f
#define SURF_BOUNDARY 0.f
#define TOP_CLAMP     1000.f
#define BOTTOM_CLAMP  (-TOP_CLAMP)

/******************************************************************************
 *
 * Multires TSDF voxel traits and algorithm specificic defines
 *
****************************************************************************/

struct MultiresSDF {
  typedef struct {
    float x;
    float x_last;
    int   y;
    int   delta_y;
  } VoxelData;

  static inline VoxelData empty()     { return {1.f, 1.f, 0, 0}; }
  static inline VoxelData initValue() { return {1.f, 1.f, 0, 0}; }
};

#endif
