# supereight: a fast octree library for dense SLAM

Welcome to *supereight*: a high performance template octree library and a dense
volumetric SLAM pipeline implementation.



## Project structure

* `se_core`: the main header only template octree library.
* `se_shared`: code required for `se_voxel_impl`, `se_denseslam` and `se_apps`.
* `se_voxel_impl`: classes intended to be used as template parameters for the
  `se_denseslam` pipeline. These classes define the data stored in the octree
  voxels and how new measurements are integrated into the octree.
* `se_denseslam`: the volumetric SLAM pipelines which can be compiled in a
  library and used in external projects. Notice that the pipeline API exposes
  the discrete octree map via an `std::shared_ptr`. As the map is a template
  class, it needs to be instantiated correctly. This is done by setting the
  value of the `SE_VOXEL_IMPLEMENTATION` macro to the name of one of the
  classes defined in `se_voxel_impl`. By default pipelines are created for each
  of the voxel implementations defined in `se_voxel_impl`.
* `se_apps`: front-end applications which run the `se_denseslam` pipelines on
  datasets or live camera.
* `se_tools`: related tools and scripts (e.g. for converting datasets and
  evaluating the pipeline).
* `third_party`: third party libraries.



## Building

Clone this repository using

``` bash
git clone --recurse-submodules [URL]
```

or if you already cloned without `--recurse-submodules` run

``` bash
git submodule update --init --recursive
```

### Installing the dependencies

The following packages are required to build the `se-denseslam` library:

* CMake >= 3.5
* Eigen3
* OpenCV
* OctoMap (optional, for OctoMap output)
* OpenMP (optional, for improved performance)
* googletest (optional, for unit tests, must be compiled from source)

The benchmarking and GUI apps additionally require:

* OpenGL
* GLut (for the simple GUI)
* PkgConfig/Qt5 (for the Qt GUI)
* OpenNI2 (for Microsoft Kinect/Asus Xtion input)
* Python2/Numpy (for the evaluation scripts)

To install the dependencies on Debian/Ubuntu run

``` bash
./se_tools/install_dependencies.sh
```

### Compiling

From the project root run

``` bash
make -j 6
```

Alternatively, use the standard steps for compiling a CMake project (the
Makefile is just a convenience wrapper):

``` bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 6
```

### Installing

After building, supereight can be optionally installed as a system-wide library
by running

``` bash
sudo make install
```

This will also install the appropriate `supereightConfig.cmake` file so that
supereight can be used in a CMake project by adding `find_package(supereight)`.
Executables using supereight should be linked against all the libraries in
`SUPEREIGHT_CORE_LIBS` and, if the dense SLAM pipeline functionality is
desired, with ONE of the libraries in the `SUPEREIGHT_DENSESLAM_LIBS` CMake
variable.  More information about the variables defined by using
`find_package(supereight)` along with usage examples can be found in
[cmake/Config.cmake.in](cmake/Config.cmake.in).

### Running the unit tests

The unit tests use googletest. It must be compiled with the same flags as
supereight and the environment variable `GTEST_ROOT` set to the path which
contains the googletest library. To compile and run the tests, assuming
googletest was compiled in `~/googletest/googletest`, run:

``` bash
GTEST_ROOT=~/googletest/googletest make test
```

### Generating the documentation

To generate the Doxygen documentation in HTML format run

``` bash
make doc
```

Then open `doc/html/index.html` with a web browser.



## Usage example

To run one of the apps in `se_apps` you need to first produce an input file. We
use [SLAMBench 1.0](https://github.com/pamela-project/slambench) file format.
The tool scene2raw can be used to produce an input sequence from the
[ICL-NUIM](http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html) dataset:

``` bash
mkdir living_room_traj2_loop
cd living_room_traj2_loop
wget http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_loop.tgz
tar xzf living_room_traj2_loop.tgz
cd ..
./build/release/se_tools/scene2raw living_room_traj2_loop living_room_traj2_loop/scene.raw
```

Then it can be used as input to one of the apps:

``` bash
./build/release/se_apps/se-denseslam-multirestsdf-pinholecamera-main --input-file living_room_traj2_loop/scene.raw --init-pose 0.34,0.5,0.3 --image-downsampling-factor 2 --integration-rate 1 --camera 481.2,-480,320,240 --map-dim 5.12 --map-size 256
```

To learn more about the command line options run:

``` bash
./build/release/se_apps/se-denseslam-multirestsdf-pinholecamera-main --help
```



## Related publications

This software implements the octree library and dense SLAM system presented in
our paper [Efficient Octree-Based Volumetric SLAM Supporting Signed-Distance
and Occupancy
Mapping.](https://spiral.imperial.ac.uk/bitstream/10044/1/55715/2/EVespaRAL_final.pdf)
It also implements the multi-resolution TSDF mapping system described in our
paper [Adaptive-Resolution Octree-Based Volumetric
SLAM.](https://www.doc.ic.ac.uk/~sleutene/publications/Vespa_3DV19.pdf) If you
publish work that relates to this software, please cite one of the following
articles:

``` bibtex
@Article{VespaRAL18,
  author  = {E. Vespa and N. Nikolov and M. Grimm and L. Nardi and P. H. J. Kelly and S. Leutenegger},
  journal = {IEEE Robotics and Automation Letters},
  title   = {Efficient Octree-Based Volumetric {SLAM} Supporting Signed-Distance and Occupancy Mapping},
  year    = {2018},
  volume  = {3},
  number  = {2},
  pages   = {1144--1151},
  month   = {Apr},
  doi     = {10.1109/LRA.2018.2792537},
}

@InProceedings{Vespa3DV19,
  author    = {E. Vespa and N. Funk and P. H. J. Kelly and S. Leutenegger},
  booktitle = {2019 International Conference on 3D Vision (3DV)},
  title     = {Adaptive-Resolution Octree-Based Volumetric {SLAM}},
  year      = {2019},
  pages     = {654--662},
  month     = {Sep},
  doi       = {10.1109/3DV.2019.00077},
}
```


## Licence

The core library is released under the BSD 3-clause licence. There are parts of
this software that are released under the MIT licence, see individual files for
which licence applies.

