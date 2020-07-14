/*
 *
 * Copyright 2016 Emanuele Vespa, Imperial College London
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * */

#include "se/voxel_implementations/OFusion/OFusion.hpp"
#include "se/str_utils.hpp"



// Initialize static data members.
constexpr bool OFusion::invert_normals;
float OFusion::surface_boundary;
float OFusion::min_occupancy;
float OFusion::max_occupancy;
float OFusion::tau;
float OFusion::sigma_min_factor;
float OFusion::sigma_max_factor;
float OFusion::sigma_min;
float OFusion::sigma_max;
float OFusion::k_sigma;

void OFusion::configure(const float voxel_dim) {
  surface_boundary = 0.f;
  min_occupancy    = -1000;
  max_occupancy    =  1000;
  tau              = 4;
  sigma_min_factor = 2;
  sigma_max_factor = 4;
  sigma_min        = sigma_min_factor * voxel_dim;
  sigma_max        = sigma_max_factor * voxel_dim;
  k_sigma          = 0.01;

}

void OFusion::configure(YAML::Node yaml_config, const float voxel_dim) {
  configure(voxel_dim);

  if (yaml_config.IsNull()) {
    return;
  }

  if (yaml_config["surface_boundary"]) {
    surface_boundary = yaml_config["surface_boundary"].as<float>();
  }
  if (yaml_config["occupancy_min_max"]) {
    std::vector<float> occupancy_min_max = yaml_config["occupancy_min_max"].as<std::vector<float>>();
    min_occupancy = occupancy_min_max[0];
    max_occupancy = occupancy_min_max[1];
  }
  if (yaml_config["tau"]) {
    tau = yaml_config["tau"].as<float>();
  }
  if (yaml_config["sigma_min_max_factor"]) {
    std::vector<float> sigma_min_max_factor = yaml_config["sigma_min_max_factor"].as<std::vector<float>>();
    sigma_min_factor = sigma_min_max_factor[0];
    sigma_max_factor = sigma_min_max_factor[1];
    sigma_min        = sigma_min_factor * voxel_dim;
    sigma_max        = sigma_max_factor * voxel_dim;
  }
  if (yaml_config["k_sigma"]) {
    k_sigma = yaml_config["k_sigma"].as<float>();
  }
}

std::string OFusion::printConfig() {

  std::stringstream out;
  out << str_utils::header_to_pretty_str("VOXEL IMPL") << "\n";
  out << str_utils::bool_to_pretty_str(OFusion::invert_normals,    "Invert normals") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::surface_boundary, "Surface boundary") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::min_occupancy,    "Min occupancy") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::max_occupancy,    "Max occupancy") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::tau,              "tau") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::sigma_min_factor, "sigma min factor") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::sigma_max_factor, "sigma max factor") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::sigma_min,        "sigma min") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::sigma_max,        "sigma max") << "\n";
  out << str_utils::value_to_pretty_str(OFusion::k_sigma,          "k sigma") << "\n";
  out << "\n";
  return out.str();
}

void OFusion::dumpMesh(OctreeType&                map,
                       std::vector<se::Triangle>& mesh) {
  auto inside = [](const VoxelData& data) {
    return data.x > OFusion::surface_boundary;
  };

  auto select_value = [](const VoxelData& data) {
    return data.x;
  };

  se::algorithms::marching_cube(map, select_value, inside, mesh);
}
