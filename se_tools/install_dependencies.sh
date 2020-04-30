#!/bin/sh
# Install the required and optional dependencies for supereight
#
# SPDX-FileCopyrightText: 2019-2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

set -eu
IFS="$(printf '%b_' '\t\n')" ; IFS="${IFS%_}"

# Required dependencies
sudo apt-get --yes install build-essential git cmake libeigen3-dev libopencv-dev

# Required source dependencies
sophus_dir='/tmp/Sophus'
rm -rf "$sophus_dir"
mkdir -p "$sophus_dir"
git clone --depth=1 https://github.com/strasdat/Sophus.git "$sophus_dir"
mkdir -p "$sophus_dir/build"
cd "$sophus_dir/build" && cmake -DCMAKE_BUILD_TYPE=Release ..
cd "$sophus_dir/build" && make && make test && sudo make install

# Optional dependencies
sudo apt-get --yes install freeglut3-dev libopenni2-dev libpapi-dev \
    qtbase5-dev python3 python3-numpy liboctomap-dev

