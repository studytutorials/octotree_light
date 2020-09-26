// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __VOLUME_HPP
#define __VOLUME_HPP

#include <Eigen/Dense>



namespace se {

  /*! \brief A cube with data.
   * Used to represent the data stored in an se::Octree at mutliple scales and
   * locations.
   */
  template <typename T>
  struct Volume {
    typedef typename T::VoxelData VoxelData;

    /** The coordinates of the Volume's centre in the Map frame. */
    Eigen::Vector3f centre_M;
    /** The lengh of the Volume's edges in metres. */
    float dim;
    /** The lengh of the Volume's edges in voxels. */
    int size;
    /** The data contained in the Volume. */
    VoxelData data;

    Volume();

    Volume(const Eigen::Vector3f& centre_M,
           float                  dim,
           int                    size,
           const VoxelData&       data);

    Volume(const Volume& other);

    ~Volume() = default;

    Volume& operator=(const Volume& other);

    bool operator==(const Volume& other) const;

    bool operator!=(const Volume& other) const;
  };

} // namespace se

#include "volume_impl.hpp"

#endif // __VOLUME_HPP

