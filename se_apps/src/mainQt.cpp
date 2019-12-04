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

#include <se/DenseSLAMSystem.h>
#include <se/perfstats.h>

#include <default_parameters.h>
#include <interface.h>
#include <PowerMonitor.h>
#ifndef __QT__
#include <draw.h>
#endif



PerfStats Stats;
PowerMonitor* powerMonitor = nullptr;
static uint16_t* input_depth = nullptr;
static uchar3* input_rgb = nullptr;
static uchar4* rgba_render = nullptr;
static uchar4* depth_render = nullptr;
static uchar4* track_render = nullptr;
static uchar4* volume_render = nullptr;
static DepthReader* reader = nullptr;
static DenseSLAMSystem* pipeline = nullptr;

static Eigen::Vector3f init_position;
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
    int                                                 frame,
    std::chrono::time_point<std::chrono::steady_clock>* timings,
    float3                                              pos,
    bool                                                tracked,
    bool                                                integrated) {

  Stats.sample("frame", frame, PerfStats::FRAME);
  Stats.sample("acquisition",  std::chrono::duration<double>(timings[1] - timings[0]).count(), PerfStats::TIME);
  Stats.sample("preprocessing",std::chrono::duration<double>(timings[2] - timings[1]).count(), PerfStats::TIME);
  Stats.sample("tracking",     std::chrono::duration<double>(timings[3] - timings[2]).count(), PerfStats::TIME);
  Stats.sample("integration",  std::chrono::duration<double>(timings[4] - timings[3]).count(), PerfStats::TIME);
  Stats.sample("raycasting",   std::chrono::duration<double>(timings[5] - timings[4]).count(), PerfStats::TIME);
  Stats.sample("rendering",    std::chrono::duration<double>(timings[6] - timings[5]).count(), PerfStats::TIME);
  Stats.sample("computation",  std::chrono::duration<double>(timings[5] - timings[1]).count(), PerfStats::TIME);
  Stats.sample("total",        std::chrono::duration<double>(timings[6] - timings[0]).count(), PerfStats::TIME);
  Stats.sample("X", pos.x, PerfStats::DISTANCE);
  Stats.sample("Y", pos.y, PerfStats::DISTANCE);
  Stats.sample("Z", pos.z, PerfStats::DISTANCE);
  Stats.sample("tracked", tracked, PerfStats::INT);
  Stats.sample("integrated", integrated, PerfStats::INT);
}

/***
 * This program loop over a scene recording
 */

int main(int argc, char** argv) {

  Configuration config = parseArgs(argc, argv);
  powerMonitor = new PowerMonitor();

  // ========= READER INITIALIZATION  =========
  reader = createReader(&config);

  //  =========  BASIC PARAMETERS  (input size / computation size )  =========
  uint2 input_size = (reader != nullptr)
      ? reader->getinputSize()
      : make_uint2(640, 480);
  const uint2 computation_size = make_uint2(
      input_size.x / config.compute_size_ratio,
      input_size.y / config.compute_size_ratio);

  //  =========  BASIC BUFFERS  (input / output )  =========

  // Construction Scene reader and input buffer
  //we could allocate a more appropriate amount of memory (less) but this makes life hard if we switch up resolution later;
  input_depth =
    (uint16_t*) malloc(sizeof(uint16_t) * input_size.x * input_size.y);
  input_rgb =
    (uchar3*) malloc(sizeof(uchar3) * input_size.x * input_size.y);
  rgba_render =
    (uchar4*) malloc(sizeof(uchar4) * computation_size.x * computation_size.y);
  depth_render =
    (uchar4*) malloc(sizeof(uchar4) * computation_size.x * computation_size.y);
  track_render =
    (uchar4*) malloc(sizeof(uchar4) * computation_size.x * computation_size.y);
  volume_render =
    (uchar4*) malloc(sizeof(uchar4) * computation_size.x * computation_size.y);

  init_position = config.initial_pos_factor.cwiseProduct(config.volume_size);
  pipeline = new DenseSLAMSystem(
      Eigen::Vector2i(computation_size.x, computation_size.y),
      Eigen::Vector3i::Constant(static_cast<int>(config.volume_resolution.x())),
      Eigen::Vector3f::Constant(config.volume_size.x()),
      init_position,
      config.pyramid, config);

  if (config.log_file != "") {
    log_file_stream.open(config.log_file.c_str());
    log_stream = &log_file_stream;
  }

  log_stream->setf(std::ios::fixed, std::ios::floatfield);

  //temporary fix to test rendering fullsize
  config.render_volume_fullsize = false;

  //The following runs the process loop for processing all the frames, if QT is specified use that, else use GLUT
  //We can opt to not run the gui which would be faster
  if (!config.no_gui) {
#ifdef __QT__
    qtLinkKinectQt(argc,argv, &pipeline, &reader, &config,
        depth_render, track_render, volume_render, rgba_render);
#else
    if ((reader == nullptr) || (reader->cameraActive == false)) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }
    while (processAll(reader, true, true, &config, false) == 0) {
      drawthem(rgba_render, depth_render, track_render, volume_render,
          track_render,
          computation_size, computation_size, computation_size, computation_size);
    }
#endif
  } else {
    if ((reader == nullptr) || (reader->cameraActive == false)) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }
    while (processAll(reader, true, true, &config, false) == 0) {
    }
    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config.dump_volume_file != "") {
    auto start = std::chrono::steady_clock::now();
    pipeline->dump_mesh(config.dump_volume_file.c_str());
    auto end = std::chrono::steady_clock::now();
    Stats.sample("meshing",
        std::chrono::duration<double>(end - start).count(),
        PerfStats::TIME);
  }

  //if (config.log_file != "") {
  //  std::ofstream logStream(config.log_file.c_str());
  //  Stats.print_all_data(logStream);
  //  logStream.close();
  //}
  //
  if (powerMonitor && powerMonitor->isActive()) {
    std::ofstream powerStream("power.rpt");
    powerMonitor->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  //powerMonitor->powerStats.print_all_data(std::cout, false);
  //std::cout << ",";
  Stats.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  //  =========  FREE BASIC BUFFERS  =========

  free(input_depth);
  free(rgba_render);
  free(depth_render);
  free(track_render);
  free(volume_render);

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
  bool raycasted = false;
  std::chrono::time_point<std::chrono::steady_clock> timings[7];
  float3 pos;
  int frame = 0;
  const uint2 input_size = (reader != nullptr)
      ? reader->getinputSize()
      : make_uint2(640, 480);
  Eigen::Vector4f camera = (reader != nullptr)
      ? reader->getK()
      : Eigen::Vector4f::Constant(0.0f);
  camera /= config->compute_size_ratio;

  if (config->camera_overrided)
    camera = config->camera / config->compute_size_ratio;

  if (reset) {
    frame_offset = reader->getFrameNumber();
  }

  if (process_frame) {
    Stats.start();
  }
  Eigen::Matrix4f pose;
  Eigen::Matrix4f gt_pose;
  timings[0] = std::chrono::steady_clock::now();
  if (process_frame) {

    // Read frames and ground truth data if set
    bool read_ok;
    if (config->groundtruth_file == "") {
      read_ok = reader->readNextDepthFrame(input_rgb, input_depth);
    } else {
      read_ok = reader->readNextData(input_rgb, input_depth, gt_pose);
    }

    // Finish processing if the next frame could not be read
    if (!read_ok) {
      timings[0] = std::chrono::steady_clock::now();
      return true;
    }

    // Process read frames
    frame = reader->getFrameNumber() - frame_offset;
    if (powerMonitor != nullptr && !first_frame)
      powerMonitor->start();

    timings[1] = std::chrono::steady_clock::now();

    pipeline->preprocessDepth(input_depth,
        Eigen::Vector2i(input_size.x, input_size.y),
        config->bilateral_filter);
    pipeline->preprocessColor((uint8_t*) input_rgb,
        Eigen::Vector2i(input_size.x, input_size.y));

    timings[2] = std::chrono::steady_clock::now();

    if (config->groundtruth_file == "") {
      // No ground truth used, call track every tracking_rate frames.
      if (frame % config->tracking_rate == 0) {
        tracked = pipeline->track(camera, config->icp_threshold);
      } else {
        tracked = false;
      }
    } else {
      // Set the pose to the ground truth.
      pipeline->setPose(gt_pose);
      tracked = true;
    }

    Eigen::Vector3f tmp = pipeline->getPosition();
    pos = make_float3(tmp.x(), tmp.y(), tmp.z());
    pose = pipeline->getPose();

    timings[3] = std::chrono::steady_clock::now();

    // Integrate only if tracking was successful every integration_rate frames
    // or it is one of the first 4 frames.
    if ((tracked && (frame % config->integration_rate == 0)) || frame <= 3) {
        integrated = pipeline->integrate(camera, config->mu, frame);
    } else {
      integrated = false;
    }

    timings[4] = std::chrono::steady_clock::now();

    if (frame > 2) {
      raycasted = pipeline->raycast(camera, config->mu);
    } else {
      raycasted = false;
    }

    timings[5] = std::chrono::steady_clock::now();
  }
  if (render_images) {
    pipeline->renderRGBA((uint8_t*) rgba_render, pipeline->getComputationResolution());
    pipeline->renderDepth((unsigned char*)depth_render, pipeline->getComputationResolution());
    pipeline->renderTrack((unsigned char*)track_render, pipeline->getComputationResolution());
    if (frame % config->rendering_rate == 0) {
      pipeline->renderVolume((unsigned char*)volume_render, pipeline->getComputationResolution(),
          camera, 0.75 * config->mu);
    }
    timings[6] = std::chrono::steady_clock::now();
  }

  if (powerMonitor != nullptr && !first_frame)
    powerMonitor->sample();

  float xt = pose(0, 3) - init_position.x();
  float yt = pose(1, 3) - init_position.y();
  float zt = pose(2, 3) - init_position.z();
  storeStats(frame, timings, pos, tracked, integrated);
  if (config->no_gui){
    *log_stream << reader->getFrameNumber() << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
  }

  //if (config->no_gui && (config->log_file == ""))
  //  Stats.print();
  first_frame = false;

  return false;
}

