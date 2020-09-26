/*

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <getopt.h>
#include <iomanip>
#include <sstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <vector>
#include <unistd.h>
#include <lodepng.h>

#include <Eigen/Dense>

#include "se/image/image.hpp"
#include "se/DenseSLAMSystem.h"
#include "se/perfstats.h"
#include "se/system_info.hpp"

#include "default_parameters.h"
#include "reader.hpp"
#include "PowerMonitor.h"
#ifdef SE_GLUT
#include "draw.h"
#endif
#include <thread>
#include <signal.h>


PowerMonitor* power_monitor = nullptr;
static uint32_t* rgba_render = nullptr;
static uint32_t* depth_render = nullptr;
static uint32_t* track_render = nullptr;
static uint32_t* volume_render = nullptr;
static se::Reader* reader = nullptr;
static DenseSLAMSystem* pipeline = nullptr;

static Eigen::Vector3f t_MW;
static std::ostream* log_stream = &std::cout;
static std::ofstream log_file_stream;

int processAll(se::Reader*        reader,
               bool               process_frame,
               bool               render_images,
               se::Configuration* config,
               bool               reset = false);

void qtLinkKinectQt(int                argc,
                    char**             argv,
                    DenseSLAMSystem**  pipeline,
                    se::Reader**       reader,
                    se::Configuration* config,
                    void*              depth_render,
                    void*              track_render,
                    void*              volume_render,
                    void*              rgba_render);

struct ProgressBar {
  ProgressBar(int total_frames=-1) : total_frames_(total_frames) {}

  void update(int curr_frame) {
    if (total_frames_ == -1) {
      // "\033[K" clear line
      std::cout << "\033[K" << "Processed frame " << std::setfill(' ') << std::setw(4)
                << curr_frame << " of whole sequence" "\n";
      // "\033[K" clear line "\r" move to beginning of line, "x1b[1A" move up one line
      std::cout << "\033[K\r\x1b[1A" << std::flush;
    } else {
      int percent = 100 * curr_frame / total_frames_;
      // "\033[K" clear line
      std::cout << "\033[K" << "Processed frame " << curr_frame << " of " << total_frames_ << "\n";
      std::stringstream ss;
      ss << std::setfill(' ') << std::setw(3) << percent;
      std::string progress = ss.str() + " % [" + std::string(percent, '*') + std::string(100 - percent, ' ') + "]";
      // "\r" move to beginning of line, "x1b[1A" move up one line
      std::cout << progress << "\r\x1b[1A" << std::flush;
    }
  }

  void end() {
    update(total_frames_);
    std::cout << "\n\n";
  }

  int total_frames_;
};

ProgressBar* progress_bar;

void storeStats(
    const size_t                                                     frame,
    const Eigen::Vector3f&                                           t_WC,
    const bool                                                       tracked,
    const bool                                                       integrated) {
  se::perfstats.sample("frame", frame, PerfStats::FRAME);
  se::perfstats.sample("t_WC.x", t_WC.x(), PerfStats::POSITION);
  se::perfstats.sample("t_WC.y", t_WC.y(), PerfStats::POSITION);
  se::perfstats.sample("t_WC.z", t_WC.z(), PerfStats::POSITION);
  se::perfstats.sample("RAM",  se::ram_usage_self() / 1024.0 / 1024.0, PerfStats::MEMORY);
  se::perfstats.sample("tracked", tracked, PerfStats::BOOL);
  se::perfstats.sample("integrated", integrated, PerfStats::BOOL);
}

/***
 * This program loop over a scene recording
 */

int main(int argc, char** argv) {

  se::Configuration config = parseArgs(argc, argv);
  power_monitor = new PowerMonitor();

  // ========= READER INITIALIZATION  =========
  reader = se::create_reader(config);
  if (reader == nullptr) {
    exit(EXIT_FAILURE);
  }

  // ========= UPDATE MAX FRAME =========
  if (config.max_frame == -1 ||
      (reader->numFrames() != 0 && config.max_frame > static_cast<long int>(reader->numFrames()) - 1)) {
    config.max_frame = reader->numFrames() - 1;
  }
  progress_bar  = new ProgressBar(config.max_frame);

  //  =========  BASIC PARAMETERS  (input image size / image size )  =========
  const Eigen::Vector2i input_image_res = (reader != nullptr)
      ? reader->depthImageRes()
      : Eigen::Vector2i(640, 480);
  const Eigen::Vector2i image_res
      = input_image_res / config.sensor_downsampling_factor;

  //  =========  BASIC BUFFERS  (input / output )  =========

  // Construction Scene reader and input buffer
  rgba_render =   new uint32_t[image_res.x() * image_res.y()];
  depth_render =  new uint32_t[image_res.x() * image_res.y()];
  track_render =  new uint32_t[image_res.x() * image_res.y()];
  volume_render = new uint32_t[image_res.x() * image_res.y()];

  t_MW = config.t_MW_factor.cwiseProduct(config.map_dim);
  pipeline = new DenseSLAMSystem(
      image_res,
      Eigen::Vector3i::Constant(config.map_size.x()),
      Eigen::Vector3f::Constant(config.map_dim.x()),
      t_MW,
      config.pyramid, config, config.voxel_impl_yaml);

  // ========= UPDATE INIT POSE =========
  se::ReaderStatus read_ok = se::ReaderStatus::ok;
  if (!config.ground_truth_file.empty()) {
    Eigen::Matrix4f init_T_WB;
    read_ok = reader->getPose(init_T_WB, 0);
    config.init_T_WB = init_T_WB;
  }
  pipeline->setInitT_WC(config.init_T_WB * config.T_BC);
  pipeline->setT_WC(config.init_T_WB * config.T_BC);

  if (read_ok != se::ReaderStatus::ok) {
    std::cerr << "Couldn't read initial pose\n";
    exit(1);
  }

  //  =========  PRINT CONFIGURATION  =========
//  config.log_file = "";
  if (config.log_file != "") {
    log_file_stream.open(config.log_file.c_str());
    log_stream = &log_file_stream;
  }
  log_stream->setf(std::ios::fixed, std::ios::floatfield);
  *log_stream << config;
  *log_stream << reader;
  *log_stream << VoxelImpl::printConfig() << std::endl;

  se::perfstats.includeDetailed(true);
  se::perfstats.setFilestream(&log_file_stream);
  //temporary fix to test rendering fullsize
  config.render_volume_fullsize = false;

#if !defined(SE_GLUT) && !defined(SE_QT)
  // Force disable render if compiled without GUI support and not in benchmark mode
  if (!config.enable_benchmark) {
    config.enable_render = false;
  }
#endif
  // The following runs the process loop for processing all the frames, if Qt
  // is specified use that, else use GLUT. We can opt to disable the gui and the rendering which
  // would be faster.
  if (config.enable_benchmark || !config.enable_render) {
    if ((reader == nullptr) || !reader->good()) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }

    while (processAll(reader, true, config.enable_render, &config, false) == 0) {}
  } else {
#ifdef SE_QT
    qtLinkKinectQt(argc,argv, &pipeline, &reader, &config,
        depth_render, track_render, volume_render, rgba_render);
#else
    if ((reader == nullptr) || !reader->good()) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }
    while (processAll(reader, true, true, &config, false) == 0) {
#ifdef SE_GLUT
      drawthem(rgba_render,   image_res,
               depth_render,  image_res,
               track_render,  image_res,
               volume_render, image_res);
#endif
    }
#endif
  }
  if (power_monitor && power_monitor->isActive()) {
    std::ofstream powerStream("power.rpt");
    power_monitor->powerStats.writeSummaryToOStream(powerStream);
    powerStream.close();
  }

  if (config.enable_benchmark) {
    progress_bar->end();
  } else {
    std::cout << "{";
    se::perfstats.writeSummaryToOStream(std::cout, false);
    std::cout << "}\n";
  }

  //  =========  FREE BASIC BUFFERS  =========
  delete pipeline;
  delete progress_bar;
  delete[] rgba_render;
  delete[] depth_render;
  delete[] track_render;
  delete[] volume_render;
}


int processAll(se::Reader*        reader,
               bool               process_frame,
               bool               render_images,
               se::Configuration* config,
               bool               reset) {
  TICK("TOTAL");
  TICK("COMPUTATION")
  static int frame_offset = 0;
  static bool first_frame = true;
  bool tracked = false;
  bool integrated = false;
  const bool track = !config->enable_ground_truth;
  const bool raycast = (track || render_images);
  int frame = 0;
  const Eigen::Vector2i input_image_res = (reader != nullptr)
      ? reader->depthImageRes()
      : Eigen::Vector2i(640, 480);
  const Eigen::Vector2i image_res
      = input_image_res / config->sensor_downsampling_factor;

  const Eigen::VectorXf elevation_angles = (Eigen::VectorXf(64) << 17.744, 17.12, 16.536, 15.982, 15.53, 14.936, 14.373, 13.823, 13.373, 12.786, 12.23, 11.687, 11.241, 10.67, 10.132, 9.574, 9.138, 8.577, 8.023, 7.479, 7.046, 6.481, 5.944, 5.395, 4.963, 4.401, 3.859, 3.319, 2.871, 2.324, 1.783, 1.238, 0.786, 0.245, -0.299, -0.849, -1.288, -1.841, -2.275, -2.926, -3.378, -3.91, -4.457, -5.004, -5.46, -6.002, -6.537, -7.096, -7.552, -8.09, -8.629, -9.196, -9.657, -10.183, -10.732, -11.289, -11.77, -12.297, -12.854, -13.415, -13.916, -14.442, -14.997, -15.595).finished();
  const Eigen::VectorXf azimuth_angles = (Eigen::VectorXf(64) << 3.102, 3.0383750000000003, 2.98175, 2.950125, 3.063, 3.021375, 3.00175, 2.996125, 3.045, 3.031375, 3.03375, 3.043125, 3.042, 3.043375, 3.05175, 3.074125, 3.03, 3.051375, 3.0797499999999998, 3.101125, 3.034, 3.067375, 3.09775, 3.142125, 3.048, 3.093375, 3.13475, 3.170125, 3.059, 3.107375, 3.15275, 3.194125, 3.085, 3.136375, 3.17675, 3.217125, 3.117, 3.159375, 3.15275, 3.257125, 3.149, 3.189375, 3.22975, 3.270125, 3.19, 3.222375, 3.26075, 3.291125, 3.23, 3.253375, 3.28775, 3.301125, 3.274, 3.299375, 3.31975, 3.306125, 3.327, 3.3453749999999998, 3.3377499999999998, 3.322125, 3.393, 3.384375, 3.35875, 3.324125).finished();
  const SensorImpl sensor({image_res.x(), image_res.y(), config->left_hand_frame,
                           config->near_plane, config->far_plane,
                           config->sensor_intrinsics[0] / config->sensor_downsampling_factor,
                           config->sensor_intrinsics[1] / config->sensor_downsampling_factor,
                           ((config->sensor_intrinsics[2] + 0.5f) / config->sensor_downsampling_factor - 0.5f),
                           ((config->sensor_intrinsics[3] + 0.5f) / config->sensor_downsampling_factor - 0.5f),
                           azimuth_angles, elevation_angles});

  static se::Image<float> input_depth_image (input_image_res.x(), input_image_res.y());
  static se::Image<uint32_t> input_rgba_image (input_image_res.x(), input_image_res.y());

  Eigen::Matrix4f T_WB;

  if (reset) {
    se::ReaderStatus read_ok = se::ReaderStatus::ok;
    frame_offset = reader->frame();
    if (!config->ground_truth_file.empty()) {
      Eigen::Matrix4f init_T_WB;
      read_ok = reader->getPose(init_T_WB, frame_offset);
      pipeline->setInitT_WC(init_T_WB * config->T_BC);
      pipeline->setT_WC(init_T_WB * config->T_BC);
    }
    if (read_ok != se::ReaderStatus::ok) {
      std::cerr << "Couldn't read pose\n";
      TOCK("COMPUTATION")
      TOCK("TOTAL")
      return true;
    }
  }

  const std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> timings (7, now);

  if (process_frame) {
    // Read frames and ground truth data if set
    se::ReaderStatus read_ok;
    TICK("ACQUISITION")
    if (config->enable_ground_truth) {
      read_ok = reader->nextData(input_depth_image, input_rgba_image, T_WB);
    } else {
      read_ok = reader->nextData(input_depth_image, input_rgba_image);
    }
    size_t reader_frame = reader->frame();
    se::perfstats.setIter(reader_frame);
    frame = reader_frame - frame_offset;
    TOCK("ACQUISITION")

    if (read_ok == se::ReaderStatus::ok) {
      // Continue normally
    } else if (read_ok == se::ReaderStatus::skip) {
      // Skip this frame
      TOCK("COMPUTATION")
      TOCK("TOTAL")
      return false;
    } else {
      // Finish processing if the next frame could not be read
      timings[0] = std::chrono::steady_clock::now();
      TOCK("COMPUTATION")
      TOCK("TOTAL")
      return true;
    }

    if (config->max_frame != -1 && frame > config->max_frame) {
      timings[0] = std::chrono::steady_clock::now();
      TOCK("COMPUTATION")
      TOCK("TOTAL")
      return true;
    }
    if (power_monitor != nullptr && !first_frame)
      power_monitor->start();

    TICK("PREPROCESSING")
    pipeline->preprocessDepth(input_depth_image.data(), input_image_res,
        config->bilateral_filter);
    pipeline->preprocessColor(input_rgba_image.data(), input_image_res);
    TOCK("PREPROCESSING")

    if (track) {
      // No ground truth used, call track every tracking_rate frames.
      if (frame % config->tracking_rate == 0) {
        tracked = pipeline->track(sensor, config->icp_threshold);
      } else {
        tracked = false;
      }
    } else {
      // Set the pose to the ground truth.
      pipeline->setT_WC(T_WB * config->T_BC);
      tracked = true;
    }



    // Integrate only if tracking was successful every integration_rate frames
    // or it is one of the first 4 frames.
    if ((tracked && (frame % config->integration_rate == 0)) || frame <= 3) {
      integrated = pipeline->integrate(sensor, frame);
    } else {
      integrated = false;
    }

    if (raycast && frame > 2) {
      pipeline->raycast(sensor);
    }
  }

  bool render_volume = false;
  if (render_images) {

    TICK("RENDERING")
    if (frame == config->max_frame) {
      render_volume = true;
    } else if (!(config->rendering_rate == 0)) {
      render_volume = (config->rendering_rate < 0) ?
          frame == std::abs(config->rendering_rate) : frame % config->rendering_rate == 0;
    }
    pipeline->renderRGBA(rgba_render, pipeline->getImageResolution());
    pipeline->renderDepth(depth_render, pipeline->getImageResolution(), sensor);
    pipeline->renderTrack(track_render, pipeline->getImageResolution());
    if (render_volume) {
      pipeline->renderVolume(volume_render, pipeline->getImageResolution(), sensor);
    }
    TOCK("RENDERING")
  }

  TOCK("COMPUTATION")

  if (power_monitor != nullptr && !first_frame) {
    power_monitor->sample();
  }

  //  =========  SAVE VOLUME RENDER  =========

  if (render_volume && config->output_render_file != "") {
    std::stringstream output_render_file_ss;
    output_render_file_ss << config->output_render_file << "_frame_"
                          << std::setw(4) << std::setfill('0') << frame << ".png";
    lodepng_encode32_file(output_render_file_ss.str().c_str(),
                          (unsigned char*)volume_render,
                          (pipeline->getImageResolution()).x(),
                          (pipeline->getImageResolution()).y());
  }

  // ==========     DUMP MESH      =========

  bool mesh_volume = false;
  if (config->enable_meshing) {
    if (frame == config->max_frame) {
      mesh_volume = true;
    } else if (!(config->meshing_rate == 0)) {
      mesh_volume = (config->meshing_rate < 0) ?
          frame == std::abs(config->meshing_rate) :frame % config->meshing_rate == 0;
    }
  }

  if (mesh_volume && config->output_mesh_file != "") {
    std::stringstream output_mesh_file_ss;
    output_mesh_file_ss << config->output_mesh_file << "_frame_"
                          << std::setw(4) << std::setfill('0') << frame << ".vtk";
    pipeline->dumpMesh(output_mesh_file_ss.str().c_str(), !config->enable_benchmark);
  }

  //  ===  SAVE OCTREE STRUCTURE AND SLICE ===
  bool save_structure = false;
  if (config->enable_structure && config->output_structure_file != "") {
    save_structure = true;
  }
  if (save_structure && (mesh_volume || frame == config->max_frame)) {
    std::stringstream output_structure_file_ss;
    output_structure_file_ss << config->output_structure_file << "_frame_"
                             << std::setw(4) << std::setfill('0') << frame;
    pipeline->saveStructure(output_structure_file_ss.str().c_str());
  }

  bool add_structure_stats = true;
  if (add_structure_stats) {
    size_t num_nodes;
    size_t num_blocks;
    std::vector<size_t> num_blocks_per_scale(VoxelImpl::VoxelBlockType::max_scale + 1, 0);
    pipeline->structureStats(num_nodes, num_blocks, num_blocks_per_scale);
    se::perfstats.sample("num_nodes", num_nodes, PerfStats::COUNT);
    se::perfstats.sample("num_blocks", num_blocks, PerfStats::COUNT);
    se::perfstats.sample("num_blocks_s0", num_blocks_per_scale[0], PerfStats::COUNT);
    se::perfstats.sample("num_blocks_s1", num_blocks_per_scale[1], PerfStats::COUNT);
    se::perfstats.sample("num_blocks_s2", num_blocks_per_scale[2], PerfStats::COUNT);
    se::perfstats.sample("num_blocks_s3", num_blocks_per_scale[3], PerfStats::COUNT);
  }

  const Eigen::Vector3f t_WC = pipeline->t_WC();
  storeStats(frame, t_WC, tracked, integrated);

  TOCK("TOTAL")

  if (config->enable_benchmark || !config->enable_render) {
    if (config->enable_benchmark) {
      if (frame % 10 == 0) {
        progress_bar->update(frame);
      }
    }

    if (config->log_file != "") {
      se::perfstats.writeToFilestream();
    } else {
      se::perfstats.writeToOStream(*log_stream);
    }
  }

  first_frame = false;
  return false;
}
