#include <iostream>

#include <se/octree.hpp>



struct Voxel {
  struct VoxelData {
    float x;
  };

  static inline VoxelData initData() { return {1.f}; }
  static inline VoxelData invalid() { return {0.f}; }

  using VoxelBlockType = se::VoxelBlockFull<Voxel>;
  using MemoryPoolType = se::PagedMemoryPool<Voxel>;
  template <typename ElemT>
  using MemoryBufferType = se::PagedMemoryBuffer<ElemT>;
};



int main(int argc, char** argv) {
  se::Octree<Voxel> octree;
  octree.init(64, 1.0f);
  std::cout << "Initialized octree\n"
      << "Voxel size: " << octree.voxelDim() << " m\n";
}

