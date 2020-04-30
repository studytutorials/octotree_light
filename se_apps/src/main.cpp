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

#include <Eigen/Dense>

#include "se/DenseSLAMSystem.h"
#include "se/perfstats.h"
#include "se/system_info.hpp"

#include "default_parameters.h"
#include "interface.h"
#include "PowerMonitor.h"
#ifdef SE_GLUT
#include "draw.h"
#endif



PerfStats stats;
PowerMonitor* power_monitor = nullptr;
static uint16_t* input_depth_image_data = nullptr;
static uchar3* input_rgb_image_data = nullptr;
static uint32_t* rgba_render = nullptr;
static uint32_t* depth_render = nullptr;
static uint32_t* track_render = nullptr;
static uint32_t* volume_render = nullptr;
static DepthReader* reader = nullptr;
static DenseSLAMSystem* pipeline = nullptr;

static Eigen::Vector3f t_MW;
static std::ostream* log_stream = &std::cout;
static std::ofstream log_file_stream;

DepthReader* createReader(Configuration* config,
                          std::string    filename = "");

int processAll(DepthReader*   reader,
               bool           process_frame,
               bool           render_images,
               Configuration* config,
               bool           reset = false);

void qtLinkKinectQt(int               argc,
                    char**            argv,
                    DenseSLAMSystem** pipeline,
                    DepthReader**     reader,
                    Configuration*    config,
                    void*             depth_render,
                    void*             track_render,
                    void*             volume_render,
                    void*             rgba_render);

void storeStats(
    int                                                              frame,
    std::vector<std::chrono::time_point<std::chrono::steady_clock>>& timings,
    const Eigen::Vector3f&                                           t_WC,
    bool                                                             tracked,
    bool                                                             integrated) {

  stats.sample("frame", frame, PerfStats::FRAME);
  stats.sample("acquisition",   std::chrono::duration<double>(timings[1] - timings[0]).count(), PerfStats::TIME);
  stats.sample("preprocessing", std::chrono::duration<double>(timings[2] - timings[1]).count(), PerfStats::TIME);
  stats.sample("tracking",      std::chrono::duration<double>(timings[3] - timings[2]).count(), PerfStats::TIME);
  stats.sample("integration",   std::chrono::duration<double>(timings[4] - timings[3]).count(), PerfStats::TIME);
  stats.sample("raycasting",    std::chrono::duration<double>(timings[5] - timings[4]).count(), PerfStats::TIME);
  stats.sample("rendering",     std::chrono::duration<double>(timings[6] - timings[5]).count(), PerfStats::TIME);
  stats.sample("computation",   std::chrono::duration<double>(timings[5] - timings[1]).count(), PerfStats::TIME);
  stats.sample("total",         std::chrono::duration<double>(timings[6] - timings[0]).count(), PerfStats::TIME);
  stats.sample("X", t_WC.x(), PerfStats::DISTANCE);
  stats.sample("Y", t_WC.y(), PerfStats::DISTANCE);
  stats.sample("Z", t_WC.z(), PerfStats::DISTANCE);
  stats.sample("tracked", tracked, PerfStats::INT);
  stats.sample("integrated", integrated, PerfStats::INT);
}

/***
 * This program loop over a scene recording
 */

int main(int argc, char** argv) {

  Configuration config = parseArgs(argc, argv);
  power_monitor = new PowerMonitor();

  // ========= READER INITIALIZATION  =========
  reader = createReader(&config);

  //  =========  BASIC PARAMETERS  (input image size / image size )  =========
  const Eigen::Vector2i input_image_res = (reader != nullptr)
      ? Eigen::Vector2i(reader->getInputImageResolution().x, reader->getInputImageResolution().y)
      : Eigen::Vector2i(640, 480);
  const Eigen::Vector2i image_res
      = input_image_res / config.image_downsampling_factor;

  //  =========  BASIC BUFFERS  (input / output )  =========

  // Construction Scene reader and input buffer
  input_depth_image_data =   new uint16_t[input_image_res.x() * input_image_res.y()];
  input_rgb_image_data =     new   uchar3[input_image_res.x() * input_image_res.y()];
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
      config.pyramid, config);

  if (config.log_file != "") {
    log_file_stream.open(config.log_file.c_str());
    log_stream = &log_file_stream;
  }
  log_stream->setf(std::ios::fixed, std::ios::floatfield);

  //temporary fix to test rendering fullsize
  config.render_volume_fullsize = false;

#if !defined(SE_GLUT) && !defined(__QT__)
  // Force no_gui if compiled without GUI support
  config.no_gui = true;
#endif
  // The following runs the process loop for processing all the frames, if Qt
  // is specified use that, else use GLUT. We can opt to not run the gui which
  // would be faster.
  if (config.no_gui) {
    if ((reader == nullptr) || (reader->cameraActive == false)) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }
    *log_stream << "frame\tacquisition\tpreprocessing\ttracking\tintegration"
        << "\traycasting\trendering\tcomputation\ttotal    \tRAM usage (MB)"
        << "\tX          \tY          \tZ         \ttracked   \tintegrated\n";
    while (processAll(reader, true, false, &config, false) == 0) {}
  } else {
#ifdef __QT__
    qtLinkKinectQt(argc,argv, &pipeline, &reader, &config,
        depth_render, track_render, volume_render, rgba_render);
#else
    if ((reader == nullptr) || (reader->cameraActive == false)) {
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

  // ==========     DUMP VOLUME      =========
  if (config.dump_volume_file != "") {
    const auto start = std::chrono::steady_clock::now();
    pipeline->dump_mesh(config.dump_volume_file.c_str());
    const auto end = std::chrono::steady_clock::now();
    stats.sample("meshing",
        std::chrono::duration<double>(end - start).count(),
        PerfStats::TIME);
  }

  if (power_monitor && power_monitor->isActive()) {
    std::ofstream powerStream("power.rpt");
    power_monitor->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  stats.print_all_data(std::cout, false);
  std::cout << "}\n";

  //  =========  FREE BASIC BUFFERS  =========
  delete pipeline;
  delete[] input_depth_image_data;
  delete[] input_rgb_image_data;
  delete[] rgba_render;
  delete[] depth_render;
  delete[] track_render;
  delete[] volume_render;
}

int processAll(DepthReader*   reader,
               bool           process_frame,
               bool           render_images,
               Configuration* config,
               bool           reset) {

  static int frame_offset = 0;
  static bool first_frame = true;
  bool tracked = false;
  bool integrated = false;
  const bool track = (config->groundtruth_file == "");
  const bool raycast = (track || render_images);
  int frame = 0;
  const Eigen::Vector2i input_image_res = (reader != nullptr)
      ? Eigen::Vector2i(reader->getInputImageResolution().x, reader->getInputImageResolution().y)
      : Eigen::Vector2i(640, 480);
  const Eigen::Vector2i image_res
      = input_image_res / config->image_downsampling_factor;
  const SensorImpl sensor({image_res.x(), image_res.y(), config->left_hand_frame,
                           nearPlane, farPlane, config->mu,
                           config->camera[0] / config->image_downsampling_factor,
                           config->camera[1] / config->image_downsampling_factor,
                           config->camera[2] / config->image_downsampling_factor,
                           config->camera[3] / config->image_downsampling_factor,
                           Eigen::VectorXf(0), Eigen::VectorXf(0)});

  if (reset) {
    frame_offset = reader->getFrameNumber();
  }

  if (process_frame) {
    stats.start();
  }
  Eigen::Matrix4f gt_T_WC;
  const std::chrono::time_point<std::chrono::steady_clock> now = std::chrono::steady_clock::now();
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> timings (7, now);

  if (process_frame) {

    // Read frames and ground truth data if set
    bool read_ok;
    if (config->groundtruth_file == "") {
      read_ok = reader->readNextDepthFrame(input_rgb_image_data, input_depth_image_data);
    } else {
      read_ok = reader->readNextData(input_rgb_image_data, input_depth_image_data, gt_T_WC);
      if (frame == 0) {
        pipeline->setInitT_WC(gt_T_WC);
      }
    }

    // Finish processing if the next frame could not be read
    if (!read_ok) {
      timings[0] = std::chrono::steady_clock::now();
      return true;
    }

    // Process read frames
    frame = reader->getFrameNumber() - frame_offset;
    if (power_monitor != nullptr && !first_frame)
      power_monitor->start();

    timings[1] = std::chrono::steady_clock::now();

    pipeline->preprocessDepth(input_depth_image_data, input_image_res,
        config->bilateral_filter);
    pipeline->preprocessColor((uint8_t*) input_rgb_image_data, input_image_res);

    timings[2] = std::chrono::steady_clock::now();

    if (track) {
      // No ground truth used, call track every tracking_rate frames.
      if (frame % config->tracking_rate == 0) {
        tracked = pipeline->track(sensor, config->icp_threshold);
      } else {
        tracked = false;
      }
    } else {
      // Set the pose to the ground truth.
      pipeline->setT_WC(gt_T_WC);
      tracked = true;
    }

    timings[3] = std::chrono::steady_clock::now();

    // Integrate only if tracking was successful every integration_rate frames
    // or it is one of the first 4 frames.
    if ((tracked && (frame % config->integration_rate == 0)) || frame <= 3) {
      integrated = pipeline->integrate(sensor, frame);
    } else {
      integrated = false;
    }

    timings[4] = std::chrono::steady_clock::now();

    if (raycast && frame > 2) {
      pipeline->raycast(sensor);
    }

    timings[5] = std::chrono::steady_clock::now();
  }
  if (render_images) {
    pipeline->renderRGBA((uint8_t*) rgba_render, pipeline->getImageResolution());
    pipeline->renderDepth((unsigned char*)depth_render, pipeline->getImageResolution(), sensor);
    pipeline->renderTrack((unsigned char*)track_render, pipeline->getImageResolution());
    if (frame % config->rendering_rate == 0) {
      pipeline->renderVolume((unsigned char*)volume_render, pipeline->getImageResolution(), sensor);
    }
  }
  timings[6] = std::chrono::steady_clock::now();

  if (power_monitor != nullptr && !first_frame)
    power_monitor->sample();

  const Eigen::Vector3f t_WC = pipeline->t_WC();
  storeStats(frame, timings, t_WC, tracked, integrated);

  if (config->no_gui) {
    const Eigen::Vector3f t_MC = pipeline->t_MC();
    *log_stream << frame << "\t"
        << std::chrono::duration<double>(timings[1] - timings[0]).count() << "\t" // acquisition
        << std::chrono::duration<double>(timings[2] - timings[1]).count() << "\t" // preprocessing
        << std::chrono::duration<double>(timings[3] - timings[2]).count() << "\t" // tracking
        << std::chrono::duration<double>(timings[4] - timings[3]).count() << "\t" // integration
        << std::chrono::duration<double>(timings[5] - timings[4]).count() << "\t" // raycasting
        << std::chrono::duration<double>(timings[6] - timings[5]).count() << "\t" // rendering
        << std::chrono::duration<double>(timings[5] - timings[1]).count() << "\t" // computation
        << std::chrono::duration<double>(timings[6] - timings[0]).count() << "\t" // total
        << se::ram_usage_self() / 1024.0 / 1024.0 << "\t" // RAM usage (MB)
        << t_MC.x() << "\t" << t_MC.y() << "\t" << t_MC.z() << "\t" // position
        << tracked << "\t" << integrated // tracked and integrated flags
        << std::endl;
  }

  first_frame = false;

  return false;
}

