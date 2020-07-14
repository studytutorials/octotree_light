# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from _benchmark_base import *
from datasets import *

FULL_BENCHMARK = BenchmarkBase("full-benchmark")
FULL_BENCHMARK.setup_from_yaml("./config/benchmark/benchmark_base_config.yaml")
FULL_BENCHMARK.add_datasets([EXAMPLE_DATASET_1, EXAMPLE_DATASET_2])
# FULL_BENCHMARK.add_datasets([ICL_NUIM, COW_AND_LADY])
