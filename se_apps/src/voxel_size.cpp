// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou, Imperial College London
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#include <se/octree_iterator.hpp>
#include <se/voxel_implementations/MultiresTSDF/MultiresTSDF.hpp>
#include <se/voxel_implementations/OFusion/OFusion.hpp>
#include <se/voxel_implementations/TSDF/TSDF.hpp>



// Use an X macro to avoid duplicating code
// https://en.wikipedia.org/wiki/X_Macro
#define VOXEL_IMPLS \
  X(TSDF) \
  X(OFusion) \
  X(MultiresTSDF) \
  X(MultiresOFusion)

// The length of the longest voxel implementation name, used for alignment when
// printing
static const int name_len = strlen("MultiresOFusion");
// The length of the longest table header, used for alignment when printing
static const int header_len = strlen("VoxelData");



// Create a function that contains a generated print statement for each voxel
// implementation
void print_sizes() {
// Define the X macro to replace each voxel implementation name with a printf
// showing its name and relevant sizes
#define X(NAME) \
  printf("%-*s %*lu B %*lu B\n", \
    name_len, #NAME, \
    header_len - 2, sizeof(NAME::VoxelData), \
    header_len - 2, sizeof(se::Volume<NAME>));

  // Evaluate the X macro for each voxel implementation
  VOXEL_IMPLS

// Remove the definition of the X macro to avoid accidental evaluations
// elsewhere in the code
#undef X
}



int main() {
  // Print the header
  printf("%-*s %*s %*s\n",
      name_len, "VoxelImpl",
      header_len, "VoxelData",
      header_len, "Volume");

  // Print the separator
  const int num_columns = name_len + 2 * header_len + 2;
  for (int i = 0; i < num_columns; i++) {
    printf("-");
  }
  printf("\n");

  // Print the sizes for each voxel implementation
  print_sizes();
}

