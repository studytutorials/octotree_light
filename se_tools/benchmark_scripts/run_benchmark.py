#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from benchmark import *
from _run import *
from systemsettings import *
from _progress_bar import *
from ruamel.yaml import YAML
yaml = YAML()

RESULTS_DIR = ""
benchmark_test_cases = FULL_BENCHMARK.generate_test_cases(RESULTS_DIR)

num_benchmark_test_cases = len(benchmark_test_cases)
btc_prog_bar = ProgressBar(100, num_benchmark_test_cases, "benchmark test case", 2)
btc_prog_bar.start()
btc_prog_bar.flash()
algorithm = Supereight(BIN_PATH)
for btc_idx, benchmark_test_case in enumerate(benchmark_test_cases):
    btc_prog_bar.update(btc_idx + 1, benchmark_test_case)
    btc_prog_bar.jump_main_to_sub_plot()
    algorithm.run(benchmark_test_case)
    btc_prog_bar.jump_sub_to_main_plot()
    btc_prog_bar.flash()
btc_prog_bar.end()
