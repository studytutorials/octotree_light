/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef DEFAULT_PARAMETERS_H_
#define DEFAULT_PARAMETERS_H_

#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"
#include <vector>
#include <sstream>
#include <getopt.h>

#include "se/constant_parameters.h"
#include "se/config.h"
#include <Eigen/Dense>

////////////////////////// RUNTIME PARAMETERS //////////////////////

#define DEFAULT_ITERATION_COUNT 3
static const int default_iterations[DEFAULT_ITERATION_COUNT] = { 10, 5, 4 };

const float default_mu = 0.1f;
const bool default_blocking_read = false;
const int default_fps = 0;
const bool default_left_hand_frame = false;
const float default_icp_threshold = 1e-5;
const int default_image_downsampling_factor = 1;
const int default_integration_rate = 2;
const int default_rendering_rate = 4;
const int default_tracking_rate = 1;
const Eigen::Vector3i default_map_size(256, 256, 256);
const Eigen::Vector3f default_map_dim(2.f, 2.f, 2.f);
const Eigen::Vector3f default_t_MW_factor(0.5f, 0.5f, 0.0f);
const bool default_no_gui = false;
const bool default_render_volume_fullsize = false;
const bool default_bilateral_filter = false;
const std::string default_dump_volume_file = "";
const std::string default_input_file = "";
const std::string default_log_file = "";
const std::string default_groundtruth_file = "";
const Eigen::Matrix4f default_gt_transform = Eigen::Matrix4f::Identity();

static std::string short_options = "a:qc:d:f:g:G:hi:l:m:k:o:p:r:s:t:v:y:z:FC:M";

static struct option long_options[] =
{
  {"block-read",         no_argument, 0, 'b'},
  {"compute-size-ratio", required_argument, 0, 'c'},
  {"dump-volume",        required_argument, 0, 'd'},
  {"fps",                required_argument, 0, 'f'},
  {"input-file",         required_argument, 0, 'i'},
  {"camera",             required_argument, 0, 'k'},
  {"icp-threshold",      required_argument, 0, 'l'},
  {"log-file",           required_argument, 0, 'o'},
  {"mu",                 required_argument, 0, 'm'},
  {"init-pose",          required_argument, 0, 'p'},
  {"no-gui",             no_argument,       0, 'q'},
  {"integration-rate",   required_argument, 0, 'r'},
  {"map-dim",            required_argument, 0, 's'},
  {"tracking-rate",      required_argument, 0, 't'},
  {"map-size",           required_argument, 0, 'v'},
  {"pyramid-levels",     required_argument, 0, 'y'},
  {"rendering-rate",     required_argument, 0, 'z'},
  {"voxel-block-size",   required_argument, 0, 'B'},
  {"bilateral-filter",   no_argument, 0, 'F'},
  {"ground-truth",       required_argument, 0, 'g'},
  {"gt-transform",       required_argument, 0, 'G'},
  {0, 0, 0, 0}
};

inline
void print_arguments() {
  std::cerr << "-b  (--block-read)                        : default is False: Block on read " << std::endl;
  std::cerr << "-c  (--image-downsampling-factor)         : default is " << default_image_downsampling_factor << "   (same size)      " << std::endl;
  std::cerr << "-d  (--dump-volume) <filename>            : Output volume file              " << std::endl;
  std::cerr << "-f  (--fps)                               : default is " << default_fps       << std::endl;
  std::cerr << "-F  (--bilateral-filter                   : default is disabled"               << std::endl;
  std::cerr << "-i  (--input-file) <filename>             : Input camera file               " << std::endl;
  std::cerr << "-k  (--camera)                            : default is defined by input     " << std::endl;
  std::cerr << "-l  (--icp-threshold)                     : default is " << default_icp_threshold << std::endl;
  std::cerr << "-o  (--log-file) <filename>               : default is stdout               " << std::endl;
  std::cerr << "-m  (--mu)                                : default is " << default_mu << "               " << std::endl;
  std::cerr << "-p  (--init-pose)                         : default is " << default_t_MW_factor.x() << "," << default_t_MW_factor.y() << "," << default_t_MW_factor.z() << "     " << std::endl;
  std::cerr << "-q  (--no-gui)                            : default is to display gui"<<std::endl;
  std::cerr << "-r  (--integration-rate)                  : default is " << default_integration_rate << "     " << std::endl;
  std::cerr << "-s  (--map-dim)                           : default is " << default_map_dim.x() << "," << default_map_dim.y() << "," << default_map_dim.z() << "      " << std::endl;
  std::cerr << "-t  (--tracking-rate)                     : default is " << default_tracking_rate << "     " << std::endl;
  std::cerr << "-v  (--map-size)                          : default is " << default_map_size.x() << "," << default_map_size.y() << "," << default_map_size.z() << "    " << std::endl;
  std::cerr << "-y  (--pyramid-levels)                    : default is 10,5,4     " << std::endl;
  std::cerr << "-z  (--rendering-rate)                    : default is " << default_rendering_rate << std::endl;
  std::cerr << "-g  (--ground-truth) <filename>           : Ground truth file" << std::endl;
  std::cerr << "-G  (--gt-transform) tx,ty,tz,qx,qy,qz,qw : Ground truth pose tranform (translation and/or rotation)" << std::endl;
}

inline Eigen::Vector3f atof3(char * optarg) {
  Eigen::Vector3f res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x() = atof(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    res.y() = res.x();
    res.z() = res.y();
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    res.z() = res.y();
  }
  return res;
}

inline Eigen::Vector3i atoi3(char * optarg) {
  Eigen::Vector3i res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x() = atoi(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y() = atoi(s.c_str());
  } else {
    res.y() = res.x();
    res.z() = res.y();
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z() = atoi(s.c_str());
  } else {
    res.z() = res.y();
  }
  return res;
}

inline Eigen::Vector4f atof4(char * optarg) {
  Eigen::Vector4f res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x() = atof(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y() = atof(s.c_str());
  } else {
    res.y() = res.x();
    res.z() = res.y();
    res.w() = res.z();
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z() = atof(s.c_str());
  } else {
    res.z() = res.y();
    res.w() = res.z();
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.w() = atof(s.c_str());
  } else {
    res.w() = res.z();
  }
  return res;
}

Configuration parseArgs(unsigned int argc, char ** argv) {

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
  config.camera_overrided = false;
  config.bilateral_filter = default_bilateral_filter;

  config.pyramid.clear();
  for (int i = 0; i < DEFAULT_ITERATION_COUNT; i++) {
    config.pyramid.push_back(default_iterations[i]);
  }

  int c;
  int option_index = 0;
  int flagErr = 0;
  std::vector<std::string> tokens;
  Eigen::Vector3f gt_transform_tran;
  Eigen::Quaternionf gt_transform_quat;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1)
    switch (c) {
      case 'a':
        {
          //float4 vals = atof4(optarg);
          //initial_pose_quant = Eigen::Quaternionf(vals.w, vals.x, vals.y, vals.z);

          //std::cerr << "update quaternion rotation to " << config.initial_pose_quant.x() << ","
          //<< config.initial_pose_quant.y() << "," << config.initial_pose_quant.z() << ","
          //<< config.initial_pose_quant.w() << std::endl;
          break;
        }
      case 'b':
        config.blocking_read = true;
        std::cerr << "activate blocking read" << std::endl;
        break;
      case 'c':  //   -c  (--image-resolution-ratio)
        config.image_downsampling_factor = atoi(optarg);
        if ((config.image_downsampling_factor != 1)
            && (config.image_downsampling_factor != 2)
            && (config.image_downsampling_factor != 4)
            && (config.image_downsampling_factor != 8)) {
          std::cerr
            << "ERROR: --image-resolution-ratio (-c) must be 1, 2 ,4 or 8  (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 'd':
        config.dump_volume_file = optarg;
        break;
      case 'e':
        //config.invert_y = true;
        //std::cerr << "Inverting Y axis (ICL-NUIM Fix)" << std::endl;
        break;

      case 'f':  //   -f  (--fps)
        config.fps = atoi(optarg);
        std::cerr << "update fps to " << config.fps << std::endl;

        if (config.fps < 0) {
          std::cerr << "ERROR: --fps (-f) must be >= 0 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 'g': // -g (--ground-truth)
        config.groundtruth_file = optarg;
        break;
      case 'G': // -G (--gt-transform)
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
            std::cerr << "Invalid number of parameters for argument gt-transform. Valid parameters are:\n"
                << "3 parameters (translation): tx,ty,tz\n"
                << "4 parameters (rotation in quaternion form): qx,qy,qz,qw\n"
                << "7 parameters (translation and rotation): tx,ty,tz,qx,qy,qz,qw"
                << std::endl;
            flagErr++;
            break;
        }
        break;
      case 'i':    //   -i  (--input-file)
        config.input_file = optarg;
        struct stat st;
        if (stat(config.input_file.c_str(), &st) != 0) {
          std::cerr << "ERROR: --input-file (-i) does not exist (was "
            << config.input_file << ")\n";
          flagErr++;
        }
        break;
      case 'k':    //   -k  (--camera)
        config.camera = atof4(optarg);
        config.camera_overrided = true;
        if (config.camera.y() < 0) {
          config.left_hand_frame = true;
          std::cerr << "update to left hand coordinate system" << std::endl;
        }
        break;
      case 'o':    //   -o  (--log-file)
        config.log_file = optarg;
        break;
      case 'l':  //   -l (--icp-threshold)
        config.icp_threshold = atof(optarg);
        break;
      case 'm':   // -m  (--mu)
        config.mu = atof(optarg);
        break;
      case 'p':    //   -p  (--init-pose)
        config.t_MW_factor = atof3(optarg);
        break;
      case 'q':
        config.no_gui = true;
        break;
      case 'r':    //   -r  (--integration-rate)
        config.integration_rate = atoi(optarg);
        if (config.integration_rate < 1) {
          std::cerr
            << "ERROR: --integration-rate (-r) must >= 1 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 's':    //   -s  (--map-size)
        config.map_dim = atof3(optarg);
        if ((config.map_dim.x() <= 0) || (config.map_dim.y() <= 0)
            || (config.map_dim.z() <= 0)) {
          std::cerr
            << "ERROR: --map-dim (-s) all dimensions must > 0 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 't':    //   -t  (--tracking-rate)
        config.tracking_rate = atof(optarg);
        break;
      case 'z':    //   -z  (--rendering-rate)
        config.rendering_rate = atof(optarg);
        break;
      case 'v':    //   -v  (--map-size)
        config.map_size = atoi3(optarg);
        if ((config.map_size.x() <= 0)
            || (config.map_size.y() <= 0)
            || (config.map_size.z() <= 0)) {
          std::cerr
            << "ERROR: --map-size (-s) all dimensions must > 0 (was "
            << optarg << ")\n";
          flagErr++;
        }

        break;
      case 'y': {
                  std::istringstream dotargs(optarg);
                  std::string s;
                  config.pyramid.clear();
                  while (getline(dotargs, s, ',')) {
                    config.pyramid.push_back(atof(s.c_str()));
                  }
                }
                break;
      case 'F':
                config.bilateral_filter = true;
                break;
      case 0:
      case '?':
                std::cerr << "Unknown option character -" << char(optopt)
                  << " or bad usage.\n";
                print_arguments();
                exit(0);
      default:
                std::cerr << "GetOpt abort.";
                flagErr = true;
    }

  if (flagErr) {
    std::cerr << "Exited due to " << flagErr << " error"
      << (flagErr == 1 ? "" : "s")
      << " in command line options\n";
    exit(1);
  }

  std::cout << config;
  return config;
}

#endif /* DEFAULT_PARAMETERS_H_ */
