# supereight Naming Conventions

This file describes the naming conventions and code style used in supereight.
It describes the desired names and for any mismatch between this document and
the code, this document is considered correct.



## Voxel

- `int voxel_C` Integer voxel coordinates, in voxel units, in frame C
- `float voxel_f_C` Float voxel coordinates, in voxel units, in frame C
- `float point_C` Float voxel coordinates, in meters, in frame C



## Pixel

- `int x, y` (width, height) to iterate over integer pixel coordinates
- `Eigen::Vector2i pixel(x,y)` to combine the above coordinates
- `float x_f, y_f` (width, height) to iterate over float pixel coordinates
- `Eigen::Vector2f pixel_f(x + 0.5f, y + 0.5f)` to combine the above coordinates
- Notes: uv coordinates in graphics are in the interval [0, 1], so they should
  only be used if that’s the case. Which is probably nowhere in supereight.



## Image

- `depth_image` the pixels should be either `float` meters (preferred) or
  `uint16_t` combined with some scaling factor (e.g. 1000 for millimeters)
- `depth_value` the value of a single pixel of the depth image
- `Eigen::Vector2i depth_image_res (width, height)`
- `rgb_image` for 3 channel color images. Pointers to RGB image data should be
  of type `uint8_t*`.
- `rgba_image` for 4 channel color images. Pointers to RGBA image data should
  be of type `uint32_t*`.




## Octree

- `voxel_depth` the depth at which single voxels are located (starts from 0)
- `block_depth` the depth at which single blocks are located (starts from 0)
- `max_morton_depth` the maximum depth the Morton codes can encode given their
  finite size (starts from 0)
- `depth`: the distance between an octree node and the root (root: 0, root's
  children 1, etc.).
- `level`: `depth + 1` (root: 1, root's children 2, etc.).
- `scale`: like `depth` but starting from the bottom and including mipmaps
  (voxel: 0, 2x2x2 voxels: 1, 4x4x4 voxels: 2, VoxelBlock: 3, VoxelBlock parent
  Nodes: 4, etc.).
- `se::Octree` is called `octree` in se_core and se_voxel_impl and `map` in
  se_denseslam.
- `size` the edge length in voxels of an octree node. For the root node this is
  the edge length of the whole octree. The number of voxels in the whole octree
  is `size * size * size`.
- `dim` the edge length in meters of an octree node. For the root node this is
  the edge length of the whole octree. The volume in cubic meters of the whole
  octree is `dim * dim * dim`.
- `voxel_dim` side length of a voxel in meter units
- Use `node` for both Nodes and pointers to nodes. Don't use `n` or `octant`.
- Use `block` for both VoxelBlocks and pointers to VoxelBlocks. Don't use `b`
  or `octant`.
- Use `idx` for the child index (0 to 7). Don't use `i` or `offset`.



## DenseSLAM

- `point(s)` a single or a bunch of 3D points with no structure
- `near_plane/far_plane` minimum and maximum sensor depth
- `T_WC` pose of the camera
- `R_WC` rotation of the camera
- `t_WC` translation of the camera
- use `pos` for position
- use `dir` for direction
- `corner_offset` absolute offset to the corners of a voxel / node
- `centre_offset` fractional offset to the centre of a voxel / node


## Other names

- Use `coord` for coordinates
- Use  `allocation_list` instead of `alloc_list`
- Use `*_value` for something that is a scalar, `*_data` otherwise.



## Code Style

- Functions
    - `int free_function(int* p, int& q);`
    - `int memberFunction(int* p, int& q);`
- Variables
    - `int local_variable;`
    - `int member_variable_;`

