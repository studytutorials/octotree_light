/*

Copyright 2016 Emanuele Vespa, Imperial College London
Copyright 2011-2020 Hauke Strasdat
Copyright 2012-2020 Steven Lovegrove

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

#ifndef __MATH_UTILS_H
#define __MATH_UTILS_H

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

/*
 * When compiling in debug mode Eigen compilation fails
 * due to -Wunused-parameter. Disable it if compiling with GCC.
 */
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if __GNUC__ > 6
#pragma GCC diagnostic ignored "-Wint-in-bool-context"
#endif
#include <Eigen/Dense>
#pragma GCC diagnostic pop
#else
#include <Eigen/Dense>
#endif



// Defining M_PI is a compiler extension, we should not rely on it.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



namespace se {
  namespace math {
    template <typename T>
    static inline T fracf(const T& v) {
        return v - v.array().floor().matrix();
      }

    template <typename T>
    static inline T floorf(const T& v) {
      return v.array().floor();
    }

    template <typename T>
    static inline T fabs(const T& v) {
      return v.cwiseAbs();
    }

    template <typename Scalar>
    static constexpr inline Scalar sq(Scalar a) {
      return a * a;
    }

    template <typename Scalar>
    static constexpr inline Scalar cu(Scalar a) {
      return a * a * a;
    }

    template <typename Scalar>
    static inline bool in(const Scalar v,
                          const Scalar a,
                          const Scalar b) {
      return v >= a && v <= b;
    }

    constexpr int log2_const(int n){
      return (n < 2 ? 0 : 1 + log2_const(n/2));
    }

    static inline Eigen::Vector3f to_translation(const Eigen::Matrix4f& T) {
      Eigen::Vector3f t = T.block<3,1>(0,3);
      return t;
    }

    static inline Eigen::Matrix3f to_rotation(const Eigen::Matrix4f& T) {
      Eigen::Matrix3f R = T.block<3,3>(0,0);
      return R;
    }

    static inline Eigen::Matrix4f to_transformation(const Eigen::Vector3f& t) {
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block<3,1>(0,3) = t;
      return T;
    }

    static inline Eigen::Matrix4f to_transformation(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block<3,3>(0,0) = R;
      T.block<3,1>(0,3) = t;
      return T;
    }

    static inline Eigen::Vector3f to_inverse_translation(const Eigen::Matrix4f& T) {
      Eigen::Vector3f t_inv = -T.block<3,3>(0,0).inverse() * T.block<3,1>(0,3);
      return t_inv;
    }

    static inline Eigen::Matrix3f to_inverse_rotation(const Eigen::Matrix4f& T) {
      Eigen::Matrix3f R_inv = (T.block<3,3>(0,0)).inverse();
      return R_inv;
    }

    static inline Eigen::Matrix4f to_inverse_transformation(const Eigen::Matrix4f& T) {
      Eigen::Matrix4f T_inv = Eigen::Matrix4f::Identity();
      T_inv.block<3,3>(0,0) = T.block<3,3>(0,0).inverse();
      T_inv.block<3,1>(0,3) = -T.block<3,3>(0,0).inverse() * T.block<3,1>(0,3);
      return T_inv;
    }

    template <typename T>
      static inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
      clamp(const T& f, const T& a, const T& b) {
        return std::max(a, std::min(f, b));
      }

    static inline void clamp(Eigen::Ref<Eigen::VectorXf>             res,
                             const Eigen::Ref<const Eigen::VectorXf> a,
                             const Eigen::Ref<Eigen::VectorXf>       b) {
      res = (res.array() < a.array()).select(a, res);
      res = (res.array() >= b.array()).select(b, res);
    }

    template <typename R, typename A, typename B>
      static inline void clamp(Eigen::MatrixBase<R>&       res,
                               const Eigen::MatrixBase<A>& a,
                               const Eigen::MatrixBase<B>& b) {
        res = res.array().max(a.array());
        res = res.array().min(b.array());
      }

    /*! \brief Compute the normal vector of a plane defined by 3 points.
     * The direction of the normal depends on the order of the points.
     */
    static Eigen::Vector4f plane_normal(const Eigen::Vector4f& p1,
                                        const Eigen::Vector4f& p2,
                                        const Eigen::Vector4f& p3) {
      // Plane tangent vectors
      const Eigen::Vector3f t1 = p2.head<3>() - p1.head<3>();
      const Eigen::Vector3f t2 = p3.head<3>() - p2.head<3>();
      // Unit normal vector
      return t1.cross(t2).normalized().homogeneous();
    }

    /*! \brief Compute the median of the data in the vector.
     *
     * \param[in,out] data The data to compute the median of. The vector will
     *                     be sorted in-place.
     * \return The median of the data. If input vector is empty, the value
     * returned by the constructor T() is returned. This is typically 0.
     *
     * \warning The vector will be sorted inside this function.
     *
     * \note Weird things will happen if the vector contains NaNs.
     */
    template <typename T>
    static T median(std::vector<T>& data) {
      if (!data.empty()) {
        std::sort(data.begin(), data.end());
        // Compute both the quotient and remainder in one go
        const std::ldiv_t result = std::ldiv(data.size(), 2);
        const size_t mid_idx = result.quot;
        if (result.rem == 0) {
          return (data[mid_idx - 1] + data[mid_idx]) / 2;
        } else {
          return data[mid_idx];
        }
      } else {
        return T();
      }
    }

    /*! \brief Compute the median of the data in the vector.
     * If the vector has an even number of elements, the second of the two
     * middle elements will be returned instead of their average. This is done
     * to avoid creating values that don't exist in the original data.
     *
     * \param[in,out] data The data to compute the median of. The vector will
     *                     be sorted in-place.
     * \return The median of the data. If input vector is empty, the value
     * returned by the constructor T() is returned. This is typically 0.
     *
     * \warning The vector will be sorted inside this function.
     *
     * \note Weird things will happen if the vector contains NaNs.
     */
    template <typename T>
    static T almost_median(std::vector<T>& data) {
      if (!data.empty()) {
        std::sort(data.begin(), data.end());
        const size_t mid_idx = data.size() / 2;
        return data[mid_idx];
      } else {
        return T();
      }
    }

    /*! Same as se::math::median() but the order of the original vector is
     * retain. This has a performance impact proportional to the size of the
     * input vector.
     */
    template <typename T>
    static T median(const std::vector<T>& data) {
      std::vector<T> v (data);
      return median(v);
    }



    /**
     * \brief hat-operator
     *
     * It takes in the 3-vector representation ``omega`` (= rotation vector) and
     * returns the corresponding matrix representation of Lie algebra element.
     *
     * Formally, the hat()-operator of SO(3) is defined as
     *
     *   ``hat(.): R^3 -> R^{3x3},  hat(omega) = sum_i omega_i * G_i``
     *   (for i=0,1,2)
     *
     * with ``G_i`` being the ith infinitesimal generator of SO(3).
     *
     * The corresponding inverse is the vee()-operator, see below.
     *
     * \param[in] omega rotation vector
     *
     * \return Corresponding matrix representation of Lie algebra element.
     */
    static Eigen::Matrix3f hat(const Eigen::Vector3f& omega) {
      Eigen::Matrix3f Omega;
      // clang-format off
      Omega <<       0.f, -omega(2),  omega(1),
          omega(2),       0.f, -omega(0),
          -omega(1),  omega(0),       0.f;
      // clang-format on
      return Omega;
    }

    static Eigen::Matrix3f exp_and_theta(const Eigen::Vector3f& omega,
                                         float&                 theta) {
      using std::sqrt;
      using std::abs;
      using std::sin;
      using std::cos;

      float theta_sq = omega.squaredNorm();
      theta = std::sqrt(theta_sq);
      float half_theta = 0.5f * theta;

      float imag_factor;
      float real_factor;
      if (theta < 1e-10) {
        float theta_po4 = theta_sq * theta_sq;
        imag_factor = 0.5f - 1.0f / 48.0f * theta_sq +
                      1.0f / 3840.0f * theta_po4;
        real_factor = 1.f - 0.5f * theta_sq + 1.f / 384.f * theta_po4;
      } else {
        float sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
        real_factor = cos(half_theta);
      }

      Eigen::Quaternionf q =
          Eigen::Quaternionf(real_factor, imag_factor * omega.x(),
                             imag_factor * omega.y(), imag_factor * omega.z());
      return q.normalized().toRotationMatrix();
    }

    /**
     * \brief Group exponential
     *
     * This functions takes in an element of tangent space (= twist ``a``) and
     * returns the corresponding element of the group SE(3).
     *
     * The first three components of ``a`` represent the translational part
     * ``upsilon`` in the tangent space of SE(3), while the last three components
     * of ``a`` represents the rotation vector ``omega``.
     * To be more specific, this function computes ``expmat(hat(a))`` with
     * ``expmat(.)`` being the matrix exponential and ``hat(.)`` the hat-operator
     * of SE(3), see below.
     */
    static Eigen::Matrix4f exp(const Eigen::Matrix<float, 6, 1>& a) {
      using std::cos;
      using std::sin;
      const Eigen::Vector3f omega = a.tail<3>();

      float theta;
      const Eigen::Matrix3f so3 = se::math::exp_and_theta(omega, theta);
      const Eigen::Matrix3f Omega = se::math::hat(omega);
      const Eigen::Matrix3f Omega_sq = Omega * Omega;
      Eigen::Matrix3f V;

      if (theta < 1e-10) {
        V = so3;
        /// Note: That is an accurate expansion!
      } else {
        float theta_sq = theta * theta;
        V = (Eigen::Matrix3f::Identity() +
             (1.f - cos(theta)) / (theta_sq) * Omega +
             (theta - sin(theta)) / (theta_sq * theta) * Omega_sq);
      }

      Eigen::Matrix4f se3 = Eigen::Matrix4f::Identity();
      se3.block<3,3>(0,0) = so3;
      se3.block<3,1>(0,3) = V * a.head<3>();
      return se3;
    }
  }
}
#endif
