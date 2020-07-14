/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef CONSTANT_PARAMETERS_H_
#define CONSTANT_PARAMETERS_H_

#include <Eigen/Dense>

////////////////////////// COMPILATION PARAMETERS //////////////////////

constexpr float e_delta = 0.1f;
constexpr int gaussian_radius = 2;
constexpr float dist_threshold = 0.1f;
constexpr float normal_threshold = 0.8f;
constexpr float track_threshold = 0.15f;

constexpr float delta = 4.0f;

const Eigen::Vector3f light{1, 1, -1.0};
const Eigen::Vector3f ambient{ 0.1, 0.1, 0.1};

#endif /* CONSTANT_PARAMETERS_H_ */
