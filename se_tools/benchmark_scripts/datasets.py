# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

from _dataset import *

EXAMPLE_DATASET_1 = Dataset("example_dataset_1")
EXAMPLE_DATASET_1.setup_from_yaml("./config/datasets/example_dataset_1_config.yaml")

EXAMPLE_DATASET_2 = Dataset("example_dataset_2")
EXAMPLE_DATASET_2.setup_from_yaml("./config/datasets/example_dataset_2_config.yaml")

# ICL_NUIM   = Dataset("icl_nuim")
# ICL_NUIM.setup_from_yaml("./config/datasets/icl_nuim_dataset_config.yaml")
#
# COW_AND_LADY   = Dataset("cow_and_lady")
# COW_AND_LADY.setup_from_yaml("./config/datasets/cow_and_lady_dataset_config.yaml")
