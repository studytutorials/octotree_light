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
#include <algorithm>
#include <iostream>

#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include "filesystem.hpp"

#include "se/config.h"
#include "se/constant_parameters.h"
#include "se/str_utils.hpp"
#include "se/utils/math_utils.h"

// Put colons after options with arguments
static std::string short_options = "bB:c:df:FgGhl:m:M:n:N:o:qQr:s:t:uUv:V:Y:z:Z:?";

static struct option long_options[] = {
  {"disable-benchmark",          no_argument,       0, 'b'},
  {"enable-benchmark",           optional_argument, 0, 'B'},
  {"sensor-downsampling-factor", required_argument, 0, 'c'},
  {"drop-frames",                no_argument,       0, 'd'},
  {"fps",                        required_argument, 0, 'f'},
  {"bilateral-filter",           no_argument,       0, 'F'},
  {"disable-ground-truth",       no_argument,       0, 'g'},
  {"enable-ground-truth",        no_argument,       0, 'G'},
  {"help",                       no_argument,       0, 'h'},
  {"icp-threshold",              required_argument, 0, 'l'},
  {"max-frame",                  required_argument, 0, 'm'},
  {"output-mesh-path",           required_argument, 0, 'M'},
  {"near-plane",                 required_argument, 0, 'n'},
  {"far-plane",                  required_argument, 0, 'N'},
  {"log-path",                   required_argument, 0, 'o'},
  {"disable-render",             no_argument,       0, 'q'},
  {"enable-render",              no_argument,       0, 'Q'},
  {"integration-rate",           required_argument, 0, 'r'},
  {"map-dim",                    required_argument, 0, 's'},
  {"tracking-rate",              required_argument, 0, 't'},
  {"disable-meshing",            no_argument,       0, 'u'},
  {"enable-meshing",             no_argument,       0, 'U'},
  {"map-size",                   required_argument, 0, 'v'},
  {"output-render-path",         required_argument, 0, 'V'},
  {"yaml-file",                  required_argument, 0, 'Y'},
  {"rendering-rate",             required_argument, 0, 'z'},
  {"meshing-rate",               required_argument, 0, 'Z'},
  {"",                           no_argument,       0, '?'},
  {0, 0, 0, 0}
};



inline void print_arguments() {
  const se::Configuration default_config;
  std::cerr << "-b  (--disable-benchmark)                         : use to override --enable-benchmark in YAML file\n";
  std::cerr << "-B  (--enable-benchmark) <blank, =filename, =dir> : default is autogen log filename\n";
  std::cerr << "-c  (--sensor-downsampling-factor)                : default is " << default_config.sensor_downsampling_factor << " (same size)\n";
  std::cerr << "-d  (--drop-frames)                               : default is false: don't drop frames\n";
  std::cerr << "-f  (--fps)                                       : default is " << default_config.fps << "\n";
  std::cerr << "-F  (--bilateral-filter                           : default is disabled\n";
  std::cerr << "-g  (--disable-ground-truth)                      : default is true if ground truth is provided\n";
  std::cerr << "-G  (--enable-ground-truth)                       : default is true if ground truth is provided\n";
  std::cerr << "-h  (--help)                                      : show this help message\n";
  std::cerr << "-l  (--icp-threshold)                             : default is " << default_config.icp_threshold << "\n";
  std::cerr << "-m  (--max-frame)                                 : default is full dataset (-1)\n";
  std::cerr << "-M  (--output-mesh-path) <filename/dir>           : output mesh path\n";
  std::cerr << "-n  (--near-plane)                                : default is " << default_config.near_plane << "\n";
  std::cerr << "-N  (--far-plane)                                 : default is " << default_config.far_plane << "\n";
  std::cerr << "-o  (--log-path) <filename/dir>                   : default is stdout\n";
  std::cerr << "-q  (--disable-render)                            : default is to render images\n";
  std::cerr << "-Q  (--enable-render)                             : use to override --disable-render in YAML file\n";
  std::cerr << "-r  (--integration-rate)                          : default is " << default_config.integration_rate << "\n";
  std::cerr << "-s  (--map-dim)                                   : default is " << default_config.map_dim.x() << "," << default_config.map_dim.y() << "," << default_config.map_dim.z() << "\n";
  std::cerr << "-t  (--tracking-rate)                             : default is " << default_config.tracking_rate << "\n";
  std::cerr << "-u  (--disable-meshing)                           : use to override --enable-meshing in YAML file\n";
  std::cerr << "-U  (--enable-meshing)                            : default is to not generate mesh\n";
  std::cerr << "-v  (--map-size)                                  : default is " << default_config.map_size.x() << "," << default_config.map_size.y() << "," << default_config.map_size.z() << "\n";
  std::cerr << "-V  (--output-render-path) <filename/dir>         : output render path\n";
  std::cerr << "-Y  (--yaml-file)                                 : YAML file\n";
  std::cerr << "-z  (--rendering-rate)                            : default is " << default_config.rendering_rate << "\n";
  std::cerr << "-Z  (--meshing-rate)                              : default is " << default_config.meshing_rate << "\n";
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



Eigen::Matrix4f toT(std::vector<float> tokens) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  Eigen::Vector3f t;
  Eigen::Quaternionf q;
  switch (tokens.size()) {
    case 3:
      // Translation
      t = Eigen::Vector3f(tokens[0], tokens[1], tokens[2]);
      T.topRightCorner<3,1>() = t;
      break;
    case 4:
      // Rotation
      // Create a quaternion and get the equivalent rotation matrix
      q = Eigen::Quaternionf(tokens[3], tokens[0],
                                                tokens[1], tokens[2]);
      T.block<3,3>(0,0) = q.toRotationMatrix();
      break;
    case 7:
      // Translation and rotation
      t = Eigen::Vector3f(tokens[0], tokens[1], tokens[2]);
      q = Eigen::Quaternionf(tokens[6], tokens[3],
                             tokens[4], tokens[5]);
      T.topRightCorner<3,1>() = t;
      T.block<3,3>(0,0) = q.toRotationMatrix();
      break;
    case 16:
      T << tokens[0],  tokens[1],  tokens[2],  tokens[3],
           tokens[4],  tokens[5],  tokens[6],  tokens[7],
           tokens[8],  tokens[9],  tokens[10], tokens[11],
           tokens[12], tokens[13], tokens[14], tokens[15];
      break;
    default:
      std::cerr << "Error: Invalid number of parameters for argument gt-transform. Valid parameters are:\n"
                << "3  parameters (translation): tx,ty,tz\n"
                << "4  parameters (rotation in quaternion form): qx,qy,qz,qw\n"
                << "7  parameters (translation and rotation): tx,ty,tz,qx,qy,qz,qw\n"
                << "16 parameters (transformation matrix): R_11, R_12, R_13, t_1\n"
                << "                                       R_21, R_22, R_23, t_2\n"
                << "                                       R_31, R_32, R_33, t_3\n"
                << std::endl;
      exit(EXIT_FAILURE);
  }
  return T;
}

Eigen::Matrix4f toT(std::vector<std::string> tokens_s) {
  std::vector<float> tokens_f;
  for (auto& token_s : tokens_s) {
    tokens_f.push_back(stof(token_s));
  }
  return toT(tokens_f);
}


std::string to_filename(std::string s) {
  std::replace(s.begin(), s.end(), '.', '_');
  std::replace(s.begin(), s.end(), '-', '_');
  std::replace(s.begin(), s.end(), ' ', '_');
  std::transform(s.begin(), s.end(),s.begin(), ::tolower);
  return s;
}

std::string autogen_filename(se::Configuration& config, std::string type) {
  if (config.sequence_name == "") {
    std::cout << "Please provide a sequence name to autogen " << type << " filename.\n"
                 "Options: \n"
                 "  - Provide sequence name via terminal          (Type \"sequence_name\" + hit enter)\n"
                 "  - Leave blank                                 (Hit enter)\n"
                 "  - Provide full filename                       (e.g. --benchmark=\"PATH/TO/result.txt\")\n"
                 "                                                      --output-render-path=\"PATH/TO/render\")\n"
                 "  - Set sequence name via command line argument (-S \"sequence_name\")\n"
                 "  - Set sequence name via YAML file             (sequence_name: \"sequence_name\") \n\n"
                 "Provide sequence name (e.g. icl-nuim-livingroom_traj_02):" << std::endl;
    std::getline(std::cin, config.sequence_name);
  }
  std::stringstream auto_filename_ss;
  auto_filename_ss                                    << config.voxel_impl_type             <<
                                      "_"             << config.sensor_type                 <<
      ((config.sequence_name != "") ? "_"              + config.sequence_name : "")         <<
                                      "_dim_"         << config.map_dim.x()                 <<
                                      "_size_"        << config.map_size.x()                <<
                                      "_down_"        << config.sensor_downsampling_factor  <<
                                      "_"             << type;
  return to_filename(auto_filename_ss.str());
}

void generate_log_file(se::Configuration& config) {
  stdfs::path log_path = config.log_file;
  if (config.log_file == "" || !stdfs::is_directory(log_path)) {
    return;
  } else {
    log_path /= autogen_filename(config, "result") + ".txt";
    config.log_file = log_path;
  }
}

void generate_render_file(se::Configuration& config) {
  // If rendering is disabled no render can be saved.
  if (!config.enable_render) {
    return; // Render is disabled. Keep render path indepentent of content.
  }

  stdfs::path output_render_path = config.output_render_file;
  // CASE 1 - Full file path provided: If the config.output_render_file is already a file use it without modification
  // NOTE: The name will be extended by "_frame_XXXX.png" when the render is actually saved.
  if (config.output_render_file != "" && !stdfs::is_directory(output_render_path)) {
    return; // Keep custom render file path
  }
  // CASE 2.1a - Nothing provided: Check if output render directory should be autogenerated
  if (config.output_render_file == "") {
    // If benchmark is active and a log file is provided, save the render in a "/render" directory within the log directory.
    if (config.enable_benchmark && config.log_file != "") {
      stdfs::path log_file = config.log_file;
      output_render_path = log_file.parent_path() / "render";
      stdfs::create_directories(output_render_path);
    } else {
      return; // Keep render file path empty ""
    }
  } // else CASE 2.1b - Directory provided: Use the provided output render directory
  // CASE 2.2 - Extend output render path with autogenerated filename.
  // NOTE: The name will be extended by "_frame_XXXX.png" when the render is actually saved.
  output_render_path /= autogen_filename(config, "render");
  config.output_render_file = output_render_path;
}

void generate_mesh_file(se::Configuration& config) {
  // Check if meshing is enabled
  if (!config.enable_meshing) {
    return; // Meshing is disabled. Keep meshing path indepentent of content.
  }

  stdfs::path output_mesh_path = config.output_mesh_file;
  // CASE 1 - Full file path provided: If the config.output_mesh_file is already a file use it without modification
  // NOTE: The name will be extended by "_frame_XXXX.vtk" when the mesh is actually saved.
  if (config.output_mesh_file != "" && !stdfs::is_directory(output_mesh_path)) {
    return; // Keep custom meshing file path
  }
  // CASE 2.1a - Nothing provided: Check if output meshing directory should be autogenerated
  if (config.output_mesh_file == "") {
    // If benchmark is active and a log file is provided, save the mesh in a "/mesh" directory within the log directory.
    if (config.enable_benchmark && config.log_file != "") {
      stdfs::path log_file = config.log_file;
      output_mesh_path = log_file.parent_path() / "mesh";
      stdfs::create_directories(output_mesh_path);
    } else { // Don't save the mesh
      return; // Keep meshing file path empty ""
    }
  } // else CASE 2.1b - Directory provided: Use the provided output meshing directory
  // CASE 2.2 - Extend mesh path with autogenerated filename.
  // NOTE: The name will be extended by "_frame_XXXX.vtk" when the mesh is actually saved.
  output_mesh_path /= autogen_filename(config, "mesh");
  config.output_mesh_file = output_mesh_path;
}

se::Configuration parseArgs(unsigned int argc, char** argv) {
	se::Configuration config;

  // Read the settings from all supplied YAML files
  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
                          &option_index)) != -1) {
    if (c == 'Y')  {
      YAML::Node yaml_general_config = YAML::Load("");
      bool has_yaml_general_config = false;
      YAML::Node yaml_map_config = YAML::Load("");
      bool has_yaml_map_config = false;
      YAML::Node yaml_sensor_config = YAML::Load("");
      bool has_yaml_sensor_config = false;

      if (YAML::LoadFile(optarg)["general"]) {
        yaml_general_config = YAML::LoadFile(optarg)["general"];
        has_yaml_general_config = true;
      }
      if (YAML::LoadFile(optarg)["map"]) {
        yaml_map_config = YAML::LoadFile(optarg)["map"];
        has_yaml_map_config = true;
      }
      if (YAML::LoadFile(optarg)["sensor"]) {
        yaml_sensor_config = YAML::LoadFile(optarg)["sensor"];
        has_yaml_sensor_config = true;
      }

      // CONFIGURE GENERAL
      // Sequence name
      if (has_yaml_general_config && yaml_general_config["sequence_name"]) {
      config.sequence_name = yaml_general_config["sequence_name"].as<std::string>();
      }
      // Sequence type
      if (has_yaml_general_config && yaml_general_config["sequence_type"]) {
        config.sequence_type = yaml_general_config["sequence_type"].as<std::string>();
        str_utils::to_lower(config.sequence_type);
      }
      // Sequence path file or directory path
      if (has_yaml_general_config && yaml_general_config["sequence_path"]) {
        config.sequence_path = yaml_general_config["sequence_path"].as<std::string>();
      }

      // En/disable ground truth
      if (has_yaml_general_config && yaml_general_config["enable_ground_truth"]) {
        config.enable_ground_truth = yaml_general_config["enable_ground_truth"].as<bool>();
      }
      // Ground truth file path
      if (has_yaml_general_config && yaml_general_config["ground_truth_file"]) {
        config.ground_truth_file = yaml_general_config["ground_truth_file"].as<std::string>();
      }

      // Benchmark and result file or directory path
      if (has_yaml_general_config && yaml_general_config["enable_benchmark"]) {
        config.enable_benchmark = yaml_general_config["enable_benchmark"].as<bool>();
      }
      // Log path
      if (has_yaml_general_config && yaml_general_config["log_path"]) {
        config.log_file = yaml_general_config["log_path"].as<std::string>();
      }
      // En/disable render
      if (has_yaml_general_config && yaml_general_config["enable_render"]) {
        config.enable_render = yaml_general_config["enable_render"].as<bool>();
      }
      // Render path
      if (has_yaml_general_config && yaml_general_config["output_render_path"]) {
        config.output_render_file = yaml_general_config["output_render_path"].as<std::string>();
      }
      // Enable render
      if (has_yaml_general_config && yaml_general_config["enable_meshing"]) {
        config.enable_meshing = yaml_general_config["enable_meshing"].as<bool>();
      }
      // Output mesh file path
      if (has_yaml_general_config && yaml_general_config["output_mesh_path"]) {
        config.output_mesh_file = yaml_general_config["output_mesh_path"].as<std::string>();
      }


      // Integration rate
      if (has_yaml_general_config && yaml_general_config["integration_rate"]) {
        config.integration_rate = yaml_general_config["integration_rate"].as<int>();
      }
      // Tracking rate
      if (has_yaml_general_config && yaml_general_config["tracking_rate"]) {
        config.tracking_rate = yaml_general_config["tracking_rate"].as<int>();
      }
      // Meshing rate
      if (has_yaml_general_config && yaml_general_config["meshing_rate"]) {
        config.meshing_rate = yaml_general_config["meshing_rate"].as<int>();
      }
      // Rendering rate
      if (has_yaml_general_config && yaml_general_config["rendering_rate"]) {
        config.rendering_rate = yaml_general_config["rendering_rate"].as<int>();
      }
      // Frames per second
      if (has_yaml_general_config && yaml_general_config["fps"]) {
        config.fps = yaml_general_config["fps"].as<float>();
      }

      // Drop frames
      if (has_yaml_general_config && yaml_general_config["drop_frames"]) {
        config.drop_frames = yaml_general_config["drop_frames"].as<bool>();
      }
      // Max frame
      if (has_yaml_general_config && yaml_general_config["max_frame"]) {
        config.max_frame = yaml_general_config["max_frame"].as<int>();
      }

      // ICP threshold
      if (has_yaml_general_config && yaml_general_config["icp_threshold"]) {
        config.icp_threshold = yaml_general_config["icp_threshold"].as<float>();
      }
      // Render volume fullsize
      if (has_yaml_general_config && yaml_general_config["render_volume_fullsize"]) {
        config.render_volume_fullsize = yaml_general_config["render_volume_fullsize"].as<bool>();
      }
      // Bilateral filter
      if (has_yaml_general_config && yaml_general_config["bilateral_filter"]) {
        config.bilateral_filter = yaml_general_config["bilateral_filter"].as<bool>();
      }

      if (has_yaml_general_config && yaml_general_config["pyramid"]) {
        config.pyramid = yaml_general_config["pyramid"].as<std::vector<int>>();
      }

      // CONFIGURE MAP
      // Map size
      if (has_yaml_map_config && yaml_map_config["size"]) {
        config.map_size = Eigen::Vector3i::Constant(yaml_map_config["size"].as<int>());
      }
      // Map dimension
      if (has_yaml_map_config && yaml_map_config["dim"]) {
        config.map_dim = Eigen::Vector3f::Constant(yaml_map_config["dim"].as<float>());
      }
      // World to Map frame translation
      if (has_yaml_map_config && yaml_map_config["t_MW_factor"]) {
        config.t_MW_factor = Eigen::Vector3f(yaml_map_config["t_MW_factor"].as<std::vector<float>>().data());
      }


      // CONFIGURE SENSOR
      // Sensor type
      config.sensor_type = SensorImpl::type();
      // Sensor intrinsics
      if (has_yaml_sensor_config && yaml_sensor_config["intrinsics"]) {
        config.sensor_intrinsics = Eigen::Vector4f((yaml_sensor_config["intrinsics"].as<std::vector<float>>()).data());
        if (config.sensor_intrinsics.y() < 0) {
          config.left_hand_frame = true;
        }
      }
      // Sensor downsamling factor
      if (has_yaml_sensor_config && yaml_sensor_config["downsampling_factor"]) {
        config.sensor_downsampling_factor = yaml_sensor_config["downsampling_factor"].as<int>();
      }
      // Camera to Body frame transformation
      if (has_yaml_sensor_config && yaml_sensor_config["T_BC"]) {
        config.T_BC = Eigen::Matrix4f(toT(yaml_sensor_config["T_BC"].as<std::vector<float>>()));
      }
      // Initial Body pose
      if (has_yaml_sensor_config && yaml_sensor_config["init_T_WB"]) {
        config.init_T_WB = Eigen::Matrix4f(toT(yaml_sensor_config["init_T_WB"].as<std::vector<float>>()));
      }
      // Near plane
      if (has_yaml_sensor_config && yaml_sensor_config["near_plane"]) {
        config.near_plane = yaml_sensor_config["near_plane"].as<float>();
      }
      // Far plane
      if (has_yaml_sensor_config && yaml_sensor_config["far_plane"]) {
        config.far_plane = yaml_sensor_config["far_plane"].as<float>();
      }
    }
  }



  // Reset getopt_long state to start parsing from the beginning
  optind = 1;
  option_index = 0;
  std::vector<std::string> tokens;
  Eigen::Vector3f t_BC;
  Eigen::Quaternionf q_BC;
  Eigen::Vector3f init_t_WB;
  Eigen::Quaternionf init_q_WB;
  // Read all other command line options
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1) {
    switch (c) {
      case 'b': // disable-benchmark
        config.enable_benchmark = false;
        break;

      case 'B': // enable-benchmark
        config.enable_benchmark = true;
        if (optarg) {
          config.log_file = optarg;
        }
        break;

      case 'c': // sensor-downsampling-factor
        config.sensor_downsampling_factor = atoi(optarg);
        if (   (config.sensor_downsampling_factor != 1)
            && (config.sensor_downsampling_factor != 2)
            && (config.sensor_downsampling_factor != 4)
            && (config.sensor_downsampling_factor != 8)) {
          std::cerr << "Error: --sensor-downsampling-factor (-c) must be 1, 2 ,4 "
              << "or 8  (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'd': // drop-frames
        config.drop_frames = true;
        break;

      case 'f': // fps
        config.fps = atof(optarg);
        if (config.fps < 0) {
          std::cerr << "Error: --fps (-f) must be >= 0 (was " << optarg << ")\n";
          exit(EXIT_FAILURE);
        }
        break;

      case 'F': // bilateral-filter
        config.bilateral_filter = true;
        break;

      case 'g': // disable-ground-truth
        config.enable_ground_truth = false;
        break;

      case 'G': // enable-ground-truth
        config.enable_ground_truth = true;
        break;

      case '?':
      case 'h': // help
        print_arguments();
        exit(EXIT_SUCCESS);

      case 'l': // icp-threshold
        config.icp_threshold = atof(optarg);
        break;

      case 'm': // max-frame
        config.max_frame = atoi(optarg);
        break;

      case 'M': // output-mesh-file
        config.output_mesh_file = optarg;
        break;

      case 'n': // near-plane
        config.near_plane = atof(optarg);
        break;

      case 'N': // far-plane
        config.far_plane = atof(optarg);
        break;

      case 'o': // log-path
        config.log_file = optarg;
        break;

      case 'q': // disable-render
        config.enable_render = false;
        break;

      case 'Q': // enable-render
        config.enable_render = true;
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

      case 'u': // disable-meshing
        config.enable_meshing = false;
        break;

      case 'U': // enable-meshing
        config.enable_meshing = true;
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

      case 'V': // output-render-path
        config.output_render_file = optarg;
        break;

      case 'Y': // yaml-file
        break;

      case 'z': // rendering-rate
        config.rendering_rate = atof(optarg);
        break;

      case 'Z': // meshing-rate
        config.meshing_rate = atof(optarg);
        break;

      default:
        print_arguments();
        exit(EXIT_FAILURE);
    }
  }

  // Reset getopt_long state to start parsing from the beginning
  optind = 1;
  option_index = 0;
  // Read the voxel implementation settings
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
                          &option_index)) != -1) {
    if (c == 'Y')  {
      YAML::Node yaml_voxel_impl_config = YAML::Load("");
      bool has_yaml_voxel_impl_config = false;

      if (YAML::LoadFile(optarg)["voxel_impl"]) {
        yaml_voxel_impl_config = YAML::LoadFile(optarg)["voxel_impl"];
        has_yaml_voxel_impl_config = true;
      }

      // CONFIGURE VOXEL IMPL
      config.voxel_impl_type = VoxelImpl::type();
      const float voxel_dim = config.map_dim.x() / config.map_size.x();
      if (has_yaml_voxel_impl_config) {
        VoxelImpl::configure(yaml_voxel_impl_config, voxel_dim);
      } else {
        VoxelImpl::configure(voxel_dim);
      }
    }
  }

  // Ensure the parameter values are valid.
  if (config.near_plane >= config.far_plane) {
    std::cerr << "Error: Near plane must be smaller than far plane ("
              << config.near_plane << " >= " << config.far_plane << ")\n";
    exit(EXIT_FAILURE);
  }

  // Autogenerate filename if only a directory is provided
  generate_log_file(config);
  generate_render_file(config);
  generate_mesh_file(config);

  return config;
}

#endif

