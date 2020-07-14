# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from _common import *
from _config import *
from _dataset import *
import warnings
import os
from ruamel.yaml import YAML
yaml = YAML()

class BenchmarkBase:
    def __init__(self, name, benchmark_base_config_yaml_path = None):
        self._name = name
        self._datasets = []

        # Base config for benchmark - Will be overwritten by dataset param
        self._config = Config()

        if is_yaml_file(benchmark_base_config_yaml_path):
            self.setup_from_yaml(benchmark_base_config_yaml_path)

    def set_config(self, config):
        assert type(config.sensor) is Sensor
        self._config = config

    def set_general(self, general):
        self._config.general = general

    def set_map(self, map):
        self._config.map = map

    def set_sensor(self, sensor):
        assert type(sensor) is Sensor
        if sensor.intrinsics:
            warnings.warn("Sensor intrinsics can't be set in benchmark base.\n"
                          "Value overwriten to None.\n"
                          "Please define in dataset or sequence.\n")
            sensor.intrinsics = None
        self._config.sensor = sensor

    def set_voxel_impls(self, voxel_impls):
        self._config.voxel_impls = voxel_impls

    def setup_from_yaml(self, benchmark_base_config_yaml_path):
        with open(benchmark_base_config_yaml_path) as f:
            benchmark_base_config_yaml = yaml.load(f)
        self._config.setup_from_yaml(benchmark_base_config_yaml)
        if self._config.sensor.intrinsics is not None:
            warnings.warn("Sensor intrinsics can't be setup in base class. Set to 'None'.")
            self._config.sensor.intrinsics = None

    def add_datasets(self, datasets):
        self._datasets += datasets

    def generate_test_cases(self, results_dir):
        results_dir = os.path.join(results_dir, "results_" + time())
        os.mkdir(results_dir)

        self._setup_missing()

        benchmark_test_cases = []
        for dataset in self._datasets:
            benchmark_test_cases += dataset.generate_test_cases(results_dir)
        return benchmark_test_cases

    def _setup_missing(self):
        for dataset in self._datasets:
            dataset.copy_missing(self._config)
            for sequence in dataset.sequences:
                sequence.copy_missing_voxel_impls(self._config.voxel_impls)

    def print_all(self):
        print(self._name)
        print("++++++++++")
        for dataset in self._datasets:
            dataset.print_all()
            print("**********")

    def print_set(self):
        print(self._name)
        print("++++++++++")
        for dataset in self._datasets:
            dataset.print_set()
            print("**********")

    def print_missing(self):
        print(self._name)
        print("++++++++++")
        for dataset in self._datasets:
            dataset.print_missing()
            print("**********")
