# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

class TestCase:

    def __init__(self):
        self.name             = None
        self.sequence_name    = None
        self.sensor_type      = None
        self.voxel_impl       = None
        self.map_res          = None
        self.downsampling_factor      = None
        self.output_dir       = None
        self.config_yaml_path = None
        self.evaluate_ate     = None

    def __str__(self):
        return self.sequence_name + " | " + self.sensor_type + " | " + self.voxel_impl + \
               " | map res: " + str(self.map_res) + " | downsampling-factor: " + str(self.downsampling_factor)


