/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */
// #include <DenseSLAMSystem.h>
#include "se/str_utils.hpp"
#include "se/commons.h"
#include "interface.h"
#include "se/constant_parameters.h"
#include "se/config.h"
#include <stdint.h>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>

DepthReader *createReader(Configuration *config, std::string filename) {
  DepthReader *reader = NULL;
  if (filename == "")
    filename = config->input_file;
  if ((filename.length() > 4)
      && (filename.substr(filename.length() - 4, 4) == ".scf")) {
    std::cerr << "====== Opening scene configuration file " << filename
      << "\n";
    std::string line;
    std::ifstream infile(filename.c_str());
    std::vector<std::string> path = split_string(filename, '/');
    std::string rpath = "";
    if (path.size() > 1)
      for (size_t i = 0; i < path.size() - 1; i++)
        rpath = rpath + path[i] + "/";
    while (std::getline(infile, line)) {
      if (line.substr(0, 1) != "#") {
        std::vector<std::string> key_value = split_string(line, '=');

        if (key_value.size() > 1) {

          std::string key = key_value[0];
          std::transform(key.begin(), key.end(), key.begin(),
              ::tolower);
          std::vector<std::string> values = split_string(key_value[1], '\"');
          std::string value;

          if (values.size() > 1)
            value = values[1];
          else
            value = values[0];
          if (key == "map-size") {
            std::vector<std::string> res = split_string(value, ',');

            if (res.size() == 3) {
              config->map_size.x() = ::atoi(
                  res[0].c_str());
              config->map_size.y() = ::atoi(
                  res[1].c_str());
              config->map_size.z() = ::atoi(
                  res[2].c_str());
            } else {
              if (res.size() == 0)
                config->map_size.x() = 256;
              else
                config->map_size.x() = ::atoi(
                    res[0].c_str());
              config->map_size.y() = config->map_size.x();
              config->map_size.z() = config->map_size.x();
            }
            std::cout << "map-size: "
              << config->map_size.x() << "x"
              << config->map_size.y() << "x"
              << config->map_size.z() << std::endl;
            continue;
          }

          if (key == "map-dim") {
            std::vector<std::string> dims = split_string(value, ',');

            if (dims.size() == 3) {
              config->map_dim.x() = ::atof(dims[0].c_str());
              config->map_dim.y() = ::atof(dims[1].c_str());
              config->map_dim.z() = ::atof(dims[2].c_str());
            } else {
              if (dims.size() == 0)
                config->map_dim.x() = 2.0;
              else {
                config->map_dim.x() = ::atof(dims[0].c_str());
                config->map_dim.y() = config->map_dim.x();
                config->map_dim.z() = config->map_dim.x();
              }
            }
            std::cout << "map-dim: " << config->map_dim.x()
              << "x" << config->map_dim.y() << "x"
              << config->map_dim.z() << std::endl;
            continue;
          }

          if (key == "world-to-map-translation-factor") {
            std::vector<std::string> dims = split_string(value, ',');
            if (dims.size() == 3) {
              config->t_MW_factor.x() = ::atof(
                  dims[0].c_str());
              config->t_MW_factor.y() = ::atof(
                  dims[1].c_str());
              config->t_MW_factor.z() = ::atof(
                  dims[2].c_str());
              std::cout << "world-to-map-translation-factor: "
                << config->t_MW_factor.x() << ", "
                << config->t_MW_factor.y() << ", "
                << config->t_MW_factor.z()
                << std::endl;
            } else {
              std::cerr
                << "ERROR: world-to-map-translation-factor  specified with incorrect data. (was "
                << value << ") Should be \"x, y, z\""
                << std::endl;
            }
            continue;
          }
          if (key == "camera") {
            std::vector<std::string> dims = split_string(value, ',');
            if (dims.size() == 4) {
              config->camera.x() = ::atof(dims[0].c_str());
              config->camera.y() = ::atof(dims[1].c_str());
              config->camera.z() = ::atof(dims[2].c_str());
              config->camera.w() = ::atof(dims[3].c_str());
              config->camera_overrided = true;
              std::cout << "camera: " << config->camera.x() << ","
                << config->camera.y() << ","
                << config->camera.z() << ","
                << config->camera.w() << std::endl;
            } else {
              std::cerr
                << "ERROR: camera specified with incorrect data. (was "
                << value << ") Should be \"x, y, z, w\""
                << std::endl;
            }

          }

          if (key == "input-file") {
            if (value.substr(0, 1) != "/") {
              value = rpath + value;
            }
            config->input_file = value;
            filename = value;
            std::cout << "input-file: " << config->input_file
              << std::endl;
            continue;
          }
        }
      }
    }
  }

  // Create reader configuration from general configuration.
  ReaderConfiguration reader_config;
  reader_config.fps = config->fps;
  reader_config.blocking_read = config->blocking_read;
  reader_config.data_path = config->input_file;
  reader_config.groundtruth_path = config->groundtruth_file;
  reader_config.transform = config->T_BC;

  struct stat st;
  lstat(filename.c_str(), &st);

  if (filename == "") {
#ifdef DO_OPENNI
    //This is for openni from a camera
    reader = new OpenNIDepthReader(reader_config);
    if(!(reader->cameraOpen)) {
      delete reader;
      reader=NULL;
    }
#else
    reader = NULL;
#endif
  } else if (S_ISDIR(st.st_mode)) {
    // ICL-NUIM reader
    reader = new SceneDepthReader(reader_config);
  }
#ifdef DO_OPENNI
  else if(filename.substr(filename.length()-4, 4)==".oni") {
    //This is for openni from a file
    reader = new OpenNIDepthReader(reader_config);
  }
#endif
  else if (filename.substr(filename.length() - 4, 4) == ".raw") {
    // Slambench 1.0 raw reader
    reader = new RawDepthReader(reader_config);
  } else {
    std::cerr << "Unrecognised file format file not loaded\n";
    reader = NULL;
  }
  if (reader && reader->isValid() == false) {
    delete reader;
    reader = NULL;
  }
  return reader;

}

