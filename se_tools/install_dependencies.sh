#!/bin/sh
# Install the required and optional dependencies for supereight
#
# SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

set -eu
IFS="$(printf '%b_' '\t\n')" ; IFS="${IFS%_}"

# Required dependencies
sudo apt-get --yes install g++ make cmake git libeigen3-dev libopencv-dev libyaml-cpp-dev

# Optional dependencies
sudo apt-get --yes install freeglut3-dev libopenni2-dev libpapi-dev \
    qtbase5-dev python3 python3-numpy liboctomap-dev

