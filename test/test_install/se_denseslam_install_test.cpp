#include <iostream>

#include <se/DenseSLAMSystem.h>



int main(int argc, char** argv) {
  se::Configuration config;

  const Eigen::Vector2i image_size (640, 480);
  const Eigen::Vector3i map_size = Eigen::Vector3i::Constant(64);
  const Eigen::Vector3f map_dim = Eigen::Vector3f::Ones();
  const Eigen::Vector3f t_MW = Eigen::Vector3f::Constant(0.5f);

  DenseSLAMSystem pipeline (image_size, map_size, map_dim, t_MW,
      config.pyramid, config);
  const Eigen::Vector3f t_WM = pipeline.t_WM();

  std::cout << "Initialized pipeline and got t_WM\n";
}

