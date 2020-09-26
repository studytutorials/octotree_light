// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

#ifndef __VOLUME_IMPL_HPP
#define __VOLUME_IMPL_HPP



template <typename T>
se::Volume<T>::Volume()
    : centre_M(Eigen::Vector3f::Constant(-1.0f)),
      dim(0.0f),
      size(0),
      data(T::initData()) {}



template <typename T>
se::Volume<T>::Volume(const Eigen::Vector3f& _centre_M,
                      float                  _dim,
                      int                    _size,
                      const VoxelData&       _data)
    : centre_M(_centre_M),
      dim(_dim),
      size(_size),
      data(_data) {}



template <typename T>
se::Volume<T>::Volume(const se::Volume<T>& other)
    : centre_M(other.centre_M),
      dim(other.dim),
      size(other.size),
      data(other.data) {}



template <typename T>
se::Volume<T>& se::Volume<T>::operator=(const se::Volume<T>& other) {
  centre_M = other.centre_M;
  dim = other.dim;
  size = other.size;
  data = other.data;
  return *this;
}



template <typename T>
bool se::Volume<T>::operator==(const se::Volume<T>& other) const {
  return (centre_M == other.centre_M)
      && (dim == other.dim)
      && (size == other.size)
      && (data == other.data);
}



template <typename T>
bool se::Volume<T>::operator!=(const se::Volume<T>& other) const {
  return !(*this == other);
}

#endif // __VOLUME_IMPL_HPP

