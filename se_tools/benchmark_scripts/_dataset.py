# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from os.path import expanduser
from _config import *
from _sequence import *
import warnings
import os
from ruamel.yaml import YAML
yaml = YAML()

class Dataset:
    def __init__(self, dataset_header):
        self.name       = None
        self.sequences  = []
        # General param for dataset - Will be overwritten by sequence param
        self.config     = Config()

        self._sequence_names = []

        if is_yaml_file(dataset_header):
            self.setup_from_yaml(dataset_header)
        elif is_instance(dataset_header, str):
            self.name = dataset_header
        else:
            warnings.warn("Use of invalid sequence header. Either provide yaml config file path or dataset name")

    def setup_from_yaml(self, dataset_config_yaml_path):
        with open(dataset_config_yaml_path) as f:
            dataset_config_yaml = yaml.load(f)
        self.config.setup_from_yaml(dataset_config_yaml)
        if 'dataset_name' not in dataset_config_yaml['general'] or \
            dataset_config_yaml['general']['dataset_name'] is None or \
            dataset_config_yaml['general']['dataset_name'] == "":
            warnings.warn("Dataset setup YAML does not contain a dataset name. Using '{}' as dataset name.".format(self.name))
        else:
            self.name = dataset_config_yaml['general']['dataset_name']
        if 'sequences' in dataset_config_yaml['general']:
            dataset_config_yaml['general']['sequences'] = list_sequences(dataset_config_yaml['general']['sequences'])
            for sequence_header in dataset_config_yaml['general']['sequences']:
                SEQUENCE = sequence_from_header(sequence_header, self._sequence_names)
                if SEQUENCE:
                    self.add_sequences([SEQUENCE])
                    self._sequence_names.append(sequence_header[0])

    def add_sequences(self, sequences):
        self.sequences += sequences

    def generate_test_cases(self, results_dir):
        dataset_dir = os.path.join(results_dir, self.name.replace('-', '_'))
        os.mkdir(dataset_dir)

        # self.setup_missing(self)

        dataset_test_cases = []
        for sequence in self.sequences:
            dataset_test_cases += sequence.generate_test_cases(dataset_dir, self.name)
        return dataset_test_cases

    def setup_missing(self):
        for sequence in self.sequences:
            sequence.copy_missing(self.config)

    def copy_missing(self, config):
        if config.sensor: config.sensor.intrinsics = None
        if not self.config:
            self.config = config
        else:
            self.config.copy_missing(config)
        for sequence in self.sequences:
            sequence.copy_missing(self.config)
            if not sequence.config.sensor.intrinsics:
                warnings.warn("{} {} misses sensor intrinsics".format(self.name, sequence.name))

    def print_all(self):
        print(self.name)
        print("==========")
        for sequence in self.sequences:
            sequence.print_all()
            print("----------")

    def print_set(self):
        print(self.name)
        print("==========")
        for sequence in self.sequences:
            sequence.print_set()
            print("----------")

    def print_missing(self):
        print(self.name)
        print("==========")
        for sequence in self.sequences:
            sequence.print_missing()
            print("----------")
