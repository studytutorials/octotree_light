/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef __DEFAULT_PARAMETERS_H
#define __DEFAULT_PARAMETERS_H

#include <cstdlib>
#include <getopt.h>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "se/config.h"
#include "se/constant_parameters.h"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"



// Default option values.
static constexpr int default_iteration_count = 3;
static constexpr int default_iterations[default_iteration_count] = { 10, 5, 4 };
static constexpr float default_mu = 0.1f;
static constexpr bool default_blocking_read = false;
static constexpr float default_fps = 0.0f;
static constexpr bool default_left_hand_frame = false;
static constexpr float default_icp_threshold = 1e-5;
static constexpr int default_image_downsampling_factor = 1;
static constexpr int default_integration_rate = 2;
static constexpr int default_rendering_rate = 4;
static constexpr int default_tracking_rate = 1;
static const Eigen::Vector3i default_map_size(256, 256, 256);
static const Eigen::Vector3f default_map_dim(2.f, 2.f, 2.f);
static const Eigen::Vector3f default_t_MW_factor(0.5f, 0.5f, 0.0f);
static constexpr bool default_no_gui = false;
static constexpr bool default_render_volume_fullsize = false;
static constexpr bool default_bilateral_filter = false;
static const std::string default_dump_volume_file = "";
static const std::string default_input_file = "";
static const std::string default_log_file = "";
static const std::string default_groundtruth_file = "";
static const Eigen::Matrix4f default_gt_transform = Eigen::Matrix4f::Identity();
static const Eigen::Vector4f default_camera = Eigen::Vector4f::Zero();



// Put colons after options with arguments
static std::string short_options = "bc:d:f:Fg:G:hi:k:l:m:o:p:qr:s:t:v:y:z:?";

static struct option long_options[] = {
  {"block-read",                no_argument,       0, 'b'},
  {"image-downsampling-factor", required_argument, 0, 'c'},
  {"dump-volume",               required_argument, 0, 'd'},
  {"fps",                       required_argument, 0, 'f'},
  {"bilateral-filter",          no_argument,       0, 'F'},
  {"ground-truth",              required_argument, 0, 'g'},
  {"gt-transform",              required_argument, 0, 'G'},
  {"help",                      no_argument,       0, 'h'},
  {"input-file",                required_argument, 0, 'i'},
  {"camera",                    required_argument, 0, 'k'},
  {"icp-threshold",             required_argument, 0, 'l'},
  {"mu",                        required_argument, 0, 'm'},
  {"log-file",                  required_argument, 0, 'o'},
  {"init-pose",                 required_argument, 0, 'p'},
  {"no-gui",                    no_argument,       0, 'q'},
  {"integration-rate",          required_argument, 0, 'r'},
  {"map-dim",                   required_argument, 0, 's'},
  {"tracking-rate",             required_argument, 0, 't'},
  {"map-size",                  required_argument, 0, 'v'},
  {"pyramid-levels",            required_argument, 0, 'y'},
  {"rendering-rate",            required_argument, 0, 'z'},
  {"",                          no_argument,       0, '?'},
  {0, 0, 0, 0}
};



inline void print_arguments() {
  std::cerr << "-b  (--block-read)                        : default is false: don't block reading\n";
  std::cerr << "-c  (--image-downsampling-factor)         : default is " << default_image_downsampling_factor << " (same size)\n";
  std::cerr << "-d  (--dump-volume) <filename>            : output mesh file\n";
  std::cerr << "-f  (--fps)                               : default is " << default_fps << "\n";
  std::cerr << "-F  (--bilateral-filter                   : default is disabled\n";
  std::cerr << "-i  (--input-file) <filename>             : input file\n";
  std::cerr << "-k  (--camera)                            : default is defined by input\n";
  std::cerr << "-l  (--icp-threshold)                     : default is " << default_icp_threshold << "\n";
  std::cerr << "-o  (--log-file) <filename>               : default is stdout\n";
  std::cerr << "-m  (--mu)                                : default is " << default_mu << "\n";
  std::cerr << "-p  (--init-pose)                         : default is " << default_t_MW_factor.x() << "," << default_t_MW_factor.y() << "," << default_t_MW_factor.z() << "\n";
  std::cerr << "-q  (--no-gui)                            : default is to display gui\n";
  std::cerr << "-r  (--integration-rate)                  : default is " << default_integration_rate << "\n";
  std::cerr << "-s  (--map-dim)                           : default is " << default_map_dim.x() << "," << default_map_dim.y() << "," << default_map_dim.z() << "\n";
  std::cerr << "-t  (--tracking-rate)                     : default is " << default_tracking_rate << "\n";
  std::cerr << "-v  (--map-size)                          : default is " << default_map_size.x() << "," << default_map_size.y() << "," << default_map_size.z() << "\n";
  std::cerr << "-y  (--pyramid-levels)                    : default is 10,5,4\n";
  std::cerr << "-z  (--rendering-rate)                    : default is " << default_rendering_rate << "\n";
  std::cerr << "-g  (--ground-truth) <filename>           : Ground truth file\n";
  std::cerr << "-G  (--gt-transform) tx,ty,tz,qx,qy,qz,qw : T_BC (translation and/or rotation)\n";
  std::cerr << "-h  (--help)                              : show this help message\n";
}



inline Eigen::Vector3f atof3(char* arg) {
  Eigen::Vector3f res = Eigen::Vector3f::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atof(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
  }
  return res;
}



inline Eigen::Vector3i atoi3(char* arg) {
  Eigen::Vector3i res = Eigen::Vector3i::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atoi(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atoi(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atoi(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
  }
  return res;
}



inline Eigen::Vector4f atof4(char* arg) {
  Eigen::Vector4f res = Eigen::Vector4f::Zero();
  std::istringstream remaining_arg(arg);
  std::string s;
  if (std::getline(remaining_arg, s, ',')) {
    res.x() = atof(s.c_str());
  } else {
    // arg is empty
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    // arg is x
    res.y() = res.x();
    res.z() = res.x();
    res.w() = res.x();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    // arg is x,y
    res.z() = res.y();
    res.w() = res.y();
    return res;
  }
  if (std::getline(remaining_arg, s, ',')) {
    res.w() = atof(s.c_str());
  } else {
    // arg is x,y,z
    res.w() = res.z();
  }
  return res;
}



Configuration parseArgs(unsigned int argc, char** argv) {

  Configuration config;

  config.image_downsampling_factor = default_image_downsampling_factor;
  config.left_hand_frame = default_left_hand_frame;
  config.integration_rate = default_integration_rate;
  config.tracking_rate = default_tracking_rate;
  config.rendering_rate = default_rendering_rate;
  config.map_size = default_map_size;
  config.map_dim = default_map_dim;
  config.t_MW_factor = default_t_MW_factor;

  config.dump_volume_file = default_dump_volume_file;
  config.input_file = default_input_file;
  config.log_file = default_log_file;
  config.groundtruth_file = default_groundtruth_file;
  config.T_BC = default_gt_transform;

  config.mu = default_mu;
  config.fps = default_fps;
  config.blocking_read = default_blocking_read;
  config.icp_threshold = default_icp_threshold;
  config.no_gui = default_no_gui;
  config.render_volume_fullsize = default_render_volume_fullsize;
  config.camera = default_camera;
  config.camera_overrided = false;
  config.bilateral_filter = default_bilateral_filter;

  config.pyramid.clear();
  for (int i = 0; i < default_iteration_count; i++) {
    config.pyramid.push_back(default_iterations[i]);
  }

  int c;
  int option_index = 0;
  std::vector<std::string> tokens;
  Eigen::Vector3f gt_transform_tran;
  Eigen::Quaternionf gt_transform_quat;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1) {
    switch (c) {
      case 'b': // blocking-read
        config.blocking_read = true;
        break;

      case 'c': // image-downsampling-factor
        config.image_downsampling_factor = atoi(optarg);
        if (   (config.image_downsampling_factor != 1)
            && (config.image_downsampling_factor != 2)
            && (config.image_downsampling_factor != 4)
            && (config.image_downsampling_factor != 8)) {
          std::cerr << "Error: --image-resolution-ratio (-c) must be 1, 2 ,4 "
              << "or 8  (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'd': // dump-volume
        config.dump_volume_file = optarg;
        break;

      case 'f': // fps
        config.fps = atof(optarg);
        if (config.fps < 0) {
          std::cerr << "Error: --fps (-f) must be >= 0 (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'g': // ground-truth
        config.groundtruth_file = optarg;
        break;

      case 'G': // gt-transform
        // Split argument into substrings
        tokens = split_string(optarg, ',');
        switch (tokens.size()) {
          case 3:
            // Translation
            gt_transform_tran = Eigen::Vector3f(std::stof(tokens[0]),
                std::stof(tokens[1]), std::stof(tokens[2]));
            config.T_BC.topRightCorner<3,1>() = gt_transform_tran;
            break;
          case 4:
            // Rotation
            // Create a quaternion and get the equivalent rotation matrix
            gt_transform_quat = Eigen::Quaternionf(std::stof(tokens[3]),
                std::stof(tokens[0]), std::stof(tokens[1]), std::stof(tokens[2]));
            config.T_BC.block<3,3>(0,0) = gt_transform_quat.toRotationMatrix();
            break;
          case 7:
            // Translation and rotation
            gt_transform_tran = Eigen::Vector3f(std::stof(tokens[0]),
                std::stof(tokens[1]), std::stof(tokens[2]));
            gt_transform_quat = Eigen::Quaternionf(std::stof(tokens[6]),
                std::stof(tokens[3]), std::stof(tokens[4]), std::stof(tokens[5]));
            config.T_BC.topRightCorner<3,1>() = gt_transform_tran;
            config.T_BC.block<3,3>(0,0) = gt_transform_quat.toRotationMatrix();
            break;
          default:
            std::cerr << "Error: Invalid number of parameters for argument gt-transform. Valid parameters are:\n"
                << "3 parameters (translation): tx,ty,tz\n"
                << "4 parameters (rotation in quaternion form): qx,qy,qz,qw\n"
                << "7 parameters (translation and rotation): tx,ty,tz,qx,qy,qz,qw"
                << std::endl;
            exit(EXIT_FAILURE);
        }
        break;

      case '?':
      case 'h': // help
        print_arguments();
        exit(EXIT_SUCCESS);

      case 'i': // input-file
        config.input_file = optarg;
        struct stat st;
        if (stat(config.input_file.c_str(), &st) != 0) {
          std::cerr << "Error: --input-file (-i) does not exist (was "
              << config.input_file << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'k': // camera
        config.camera = atof4(optarg);
        config.camera_overrided = true;
        if (config.camera.y() < 0) {
          config.left_hand_frame = true;
          std::cerr << "update to left hand coordinate system" << std::endl;
        }
        break;

      case 'o': // log-file
        config.log_file = optarg;
        break;

      case 'l': // icp-threshold
        config.icp_threshold = atof(optarg);
        break;

      case 'm': // mu
        config.mu = atof(optarg);
        break;

      case 'p': // init-pose
        config.t_MW_factor = atof3(optarg);
        break;

      case 'q': // no-qui
        config.no_gui = true;
        break;

      case 'r': // integration-rate
        config.integration_rate = atoi(optarg);
        if (config.integration_rate < 1) {
          std::cerr << "Error: --integration-rate (-r) must >= 1 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 's': // map-size
        config.map_dim = atof3(optarg);
        if (   (config.map_dim.x() <= 0)
            || (config.map_dim.y() <= 0)
            || (config.map_dim.z() <= 0)) {
          std::cerr << "Error: --map-dim (-s) all dimensions must > 0 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 't': // tracking-rate
        config.tracking_rate = atof(optarg);
        break;

      case 'z': // rendering-rate
        config.rendering_rate = atof(optarg);
        break;

      case 'v': // map-size
        config.map_size = atoi3(optarg);
        if (   (config.map_size.x() <= 0)
            || (config.map_size.y() <= 0)
            || (config.map_size.z() <= 0)) {
          std::cerr << "Error: --map-size (-s) all dimensions must > 0 (was "
              << optarg << ")\n";
          exit(EXIT_FAILURE);
        }

        break;

      case 'y': // pyramid-levels
        {
          std::istringstream remaining_arg(optarg);
          std::string s;
          config.pyramid.clear();
          while (std::getline(remaining_arg, s, ',')) {
            config.pyramid.push_back(atof(s.c_str()));
          }
        }
        break;

      case 'F': // bilateral-filter
        config.bilateral_filter = true;
        break;

      default:
        print_arguments();
        exit(EXIT_FAILURE);
    }
  }

  std::cout << config;
  return config;
}

#endif

