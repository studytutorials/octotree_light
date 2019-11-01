# `se_voxel_impl`

## Adding new voxel implementations
There is a minimal example voxel implementation in
`se_voxel_impl/include/se/voxel_implementations/ExampleVoxelImpl` and
`se_voxel_impl/src/ExampleVoxelImpl`. Voxel implementations consist of at least
one header and one source file which are compiled into a static library. They
must have the same folder structure as `ExampleVoxelImpl`, with the folders and
header having the same name as the class. All source files matching the pattern
`se_voxel_impl/src/ExampleVoxelImpl/ExampleVoxelImpl*.cpp` will be included in
the resulting library. Any number of source files conforming to this pattern can
be added, as well as any number of headers.

### Example
Below is an example of a valid file/folder structure:
``` text
se_voxel_impl
├─ include
│  └─ se
│     └─ voxel_implementations
│        └─ MyVoxel
│           ├─ MyVoxel.hpp
│           ├─ MyVoxel_additional_header_1.hpp
│           └─ MyVoxel_additional_header_2.hpp
└─ src
   └─ MyVoxel
      ├─ MyVoxel.cpp
      ├─ MyVoxel_additional_source_1.cpp
      └─ MyVoxel_additional_source_2.cpp
```

Additionally, in `se_voxel_impl/CMakeLists.txt` `MyVoxel` should be appended to
`SUPEREIGHT_VOXEL_IMPLS`. Finally, the line `#include
<se/voxel_implementations/MyVoxel/MyVoxel.hpp>` must be added to
`se_denseslam/include/se/voxel_implementations.hpp`. This is enough for
compiling the pipeline in `se_denseslam` using `MyVoxel` as the template
parameter.

