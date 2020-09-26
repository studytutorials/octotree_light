#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause

# TODO allow plotting data as it arrives through a pipe so we can do
#   supereight -i ... | ./plot_stats.py
# and see live stats

import argparse
import fileinput
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np



class PerfStats:
    """Statistics for a single run of supereight"""
    def __init__(self, filename: str="") -> None:
        self.filename = filename
        self.result_header_idx = None

        file = open(filename, 'r')
        lines = file.readlines()
        # Find line idx at which the results start (i.e. line after 'RESULT DATA\n')
        for idx, line in enumerate(lines):
            if line == 'RESULT DATA\n':
                self.result_header_idx = idx + 1
                break;

        result_header = lines[self.result_header_idx].split('\t')
        self.frame_idx = result_header.index('frame [#]')
        self.acquisition_time_idx = result_header.index('ACQUISITION [ds]')
        self.preprocessing_time_idx = result_header.index('PREPROCESSING [ds]')
        self.tracking_time_idx = result_header.index('TRACKING [ds]') if 'TRACKING [ds]' in result_header else None
        self.integration_time_idx = result_header.index('INTEGRATION [ds]')
        self.raycasting_time_idx = result_header.index('RAYCASTING [ds]') if 'RAYCASTING [ds]' in result_header else None
        self.rendering_time_idx = result_header.index('RENDERING [ds]') if 'RENDERING [ds]' in result_header else None
        self.computation_time_idx = result_header.index('COMPUTATION [ds]')
        self.total_time_idx = result_header.index('TOTAL [ds]')
        self.ram_usage_idx = result_header.index('RAM [MB]')
        self.num_nodes_idx = result_header.index('num_nodes [count]') if 'num_nodes [count]' in result_header else None
        self.num_blocks_t_idx = result_header.index('num_blocks [count]') if 'num_blocks [count]' in result_header else None
        self.num_blocks_0_idx = result_header.index('num_blocks_s0 [count]') if 'num_blocks_s0 [count]' in result_header else None
        self.num_blocks_1_idx = result_header.index('num_blocks_s1 [count]') if 'num_blocks_s1 [count]' in result_header else None
        self.num_blocks_2_idx = result_header.index('num_blocks_s2 [count]') if 'num_blocks_s2 [count]' in result_header else None
        self.num_blocks_3_idx = result_header.index('num_blocks_s3 [count]') if 'num_blocks_s3 [count]' in result_header else None

        self.frames = []
        self.acquisition_time = []
        self.preprocessing_time = []
        self.tracking_time = []
        self.integration_time = []
        self.raycasting_time = []
        self.rendering_time = []
        self.computation_time = []
        self.total_time = []
        self.ram_usage = []
        self.num_nodes = []
        self.num_blocks_t= []
        self.num_blocks_0= []
        self.num_blocks_1= []
        self.num_blocks_2= []
        self.num_blocks_3= []

        self.num_subplts = 3 if self.num_nodes_idx else 2

    def append_line(self, line: str) -> None:
        # Ignore lines not starting with whitespace or digits
        if not (line[0].isspace() or line[0].isdecimal()):
            return
        # Split the line at whitespace
        columns = line.replace("\t", " ").split()

        # Append the data
        self.frames.append(
            int(columns[self.frame_idx]) if columns[self.frame_idx] != '*' else 0)
        self.acquisition_time.append(
            float(columns[self.acquisition_time_idx]) if columns[self.acquisition_time_idx] != '*' else 0)
        self.preprocessing_time.append(
            float(columns[self.preprocessing_time_idx]) if columns[self.preprocessing_time_idx] != '*' else 0)
        self.tracking_time.append(
            (float(columns[self.tracking_time_idx]) if columns[self.tracking_time_idx] != '*' else 0) if
            self.tracking_time_idx else 0)
        self.integration_time.append(
            float(columns[self.integration_time_idx]) if columns[self.integration_time_idx] != '*' else 0)
        self.raycasting_time.append(
            (float(columns[self.raycasting_time_idx]) if columns[self.raycasting_time_idx] != '*' else 0)
            if self.raycasting_time_idx else 0)
        self.rendering_time.append(
            (float(columns[self.rendering_time_idx]) if columns[self.rendering_time_idx] != '*' else 0)
            if self.rendering_time_idx else 0)
        self.computation_time.append(
            float(columns[self.computation_time_idx]) if columns[self.computation_time_idx] != '*' else 0)
        self.total_time.append(
            float(columns[self.total_time_idx]) if columns[self.total_time_idx] != '*' else 0)
        self.ram_usage.append(
            float(columns[self.ram_usage_idx]) if columns[self.ram_usage_idx] != '*' else 0)
        self.num_nodes.append(
            (float(columns[self.num_nodes_idx]) if columns[self.num_nodes_idx] != '*' else 0) if
            self.num_nodes_idx else 0)
        self.num_blocks_t.append(
            (float(columns[self.num_blocks_t_idx]) if columns[self.num_blocks_t_idx] != '*' else 0) if
            self.num_blocks_t_idx else 0)
        self.num_blocks_0.append(
            (float(columns[self.num_blocks_0_idx]) if columns[self.num_blocks_0_idx] != '*' else 0) if
            self.num_blocks_0_idx else 0)
        self.num_blocks_1.append(
            (float(columns[self.num_blocks_1_idx]) if columns[self.num_blocks_1_idx] != '*' else 0) if
            self.num_blocks_1_idx else 0)
        self.num_blocks_2.append(
            (float(columns[self.num_blocks_2_idx]) if columns[self.num_blocks_2_idx] != '*' else 0) if
            self.num_blocks_2_idx else 0)
        self.num_blocks_3.append(
            (float(columns[self.num_blocks_3_idx]) if columns[self.num_blocks_3_idx] != '*' else 0) if
            self.num_blocks_3_idx else 0)

    def last_frame(self) -> 'PerfStats':
        # Create an PerfStats object containing the data of the last frame
        d = PerfStats(self.filename)
        if self.frames:
            d.frames.append(self.frames[-1])
            d.acquisition_time.append(self.acquisition_time[-1])
            d.preprocessing_time.append(self.preprocessing_time[-1])
            d.tracking_time.append(self.tracking_time[-1])
            d.integration_time.append(self.integration_time[-1])
            d.raycasting_time.append(self.raycasting_time[-1])
            d.rendering_time.append(self.rendering_time[-1])
            d.computation_time.append(self.computation_time[-1])
            d.total_time.append(self.total_time[-1])
            d.ram_usage.append(self.ram_usage[-1])
            d.num_nodes.append(self.num_nodes[-1])
            d.num_blocks_t.append(self.num_blocks_t[-1])
            d.num_blocks_0.append(self.num_blocks_0[-1])
            d.num_blocks_1.append(self.num_blocks_1[-1])
            d.num_blocks_2.append(self.num_blocks_2[-1])
            d.num_blocks_3.append(self.num_blocks_3[-1])
        return d

    def plot(self, axes=None) -> None:
        # Create a new subplot only if an existing one wasn't provided.
        if axes is None:
            _, axes = plt.subplots(self.num_subplts, 1)
            axes = np.append(axes, axes[self.num_subplts - 1].twinx())

        # Compute the basename of the file the data came from.
        file_basename = os.path.basename(self.filename)

        timing_labels=['Acquisition', 'Preprocessing', 'Tracking',
                'Integration', 'Raycasting', 'Rendering']
        axes[0].stackplot(self.frames,
                [1000 * x for x in self.acquisition_time],
                [1000 * x for x in self.preprocessing_time],
                [1000 * x for x in self.tracking_time],
                [1000 * x for x in self.integration_time],
                [1000 * x for x in self.raycasting_time],
                [1000 * x for x in self.rendering_time],
                labels=timing_labels)
        axes[0].legend(loc='upper left')
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('Time (ms)')
        axes[0].set_title('Computation time')

        if self.num_subplts == 3:
          num_labels=['Blocks S=0', 'Blocks S=1', 'Blocks S=2',
                      'Blocks S=3', 'Blocks S=-1', ]

          num_blocks_zip = zip(self.num_blocks_t,
                               self.num_blocks_0,
                               self.num_blocks_1,
                               self.num_blocks_2,
                               self.num_blocks_3)
          num_blocks_d = [t - s0 - s1 - s2 - s3 for
                          (t, s0, s1, s2, s3) in num_blocks_zip]

          axes[1].stackplot(self.frames,
                            self.num_blocks_0,
                            self.num_blocks_1,
                            self.num_blocks_2,
                            self.num_blocks_3,
                            num_blocks_d,
                            labels=num_labels)

          axes[1].set_xlabel('Frame')
          axes[1].set_ylabel('Number')
          axes[1].set_title('Number of nodes and blocks')

          num_nodes_colour = 'tab:cyan'
          axes[1].plot(self.frames, self.num_nodes, color=num_nodes_colour, label="Nodes [total]")
          num_blocks_colour = 'tab:grey'
          axes[1].plot(self.frames, self.num_blocks_t, color=num_blocks_colour, label="Blocks [total]")
          axes[1].legend(loc='upper left')

        ram_a_idx  = self.num_subplts - 1
        time_a_idx = self.num_subplts
        ram_colour = 'tab:blue'
        axes[ram_a_idx].stackplot(self.frames, self.ram_usage, color=ram_colour)
        axes[ram_a_idx].set_xlabel('Frame')
        axes[ram_a_idx].set_ylabel('RAM (MiB)', color=ram_colour)
        axes[ram_a_idx].set_title('Resource usage')

        time_colour = 'tab:green'
        axes[time_a_idx].plot(self.frames, [1000 * x for x in self.total_time], color=time_colour)
        axes[time_a_idx].set_ylabel('Computation time (ms)', color=time_colour)
        return axes



def parse_arguments():
    parser = argparse.ArgumentParser(
            description=('Plot the statistics shown when running supereight '
                'with the --no-gui option. Run supereight and redirect the '
                'output to a file using '
                'se-denseslam-tsdf-pinholecamera-main --no-gui ... > log.txt '
                'and plot the data using '
                './se_tools/plot_stats.py log.txt'))
    parser.add_argument('files', nargs='*', metavar='FILE', default=['-'],
            help=('A text file containing the output of supereight. With no '
                'FILE or when FILE is -, read standard input. If multiple '
                'files are provided the results for all files are shown in a '
                'single window.'))
    parser.add_argument('-A', '--no-equalize-axes', action='store_false',
            dest='equalize_axes',
            help=('Don\'t equalize the vertical axes of subplots.'))
    parser.add_argument('--hide-plot', help='hide plot', action="store_false", dest="show_plot")
    parser.add_argument('--save-plot', help='save plot', nargs='?', const="./plot.png", dest="plot_file")
    args = parser.parse_args()
    return args



def equalize_y_axes(axes) -> None:
    # Equalize each row
    for row in axes:
        y_min = float('inf');
        y_max = float('-inf');
        # Find the largest y axis limits
        for ax in row:
            b, t = ax.get_ylim()
            if b < y_min:
                y_min = b
            if t > y_max:
                y_max = t
        # Apply the largest y axis limits
        for ax in row:
            #ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)



if __name__ == "__main__":
    try:
        args = parse_arguments()

        # Read the data
        data = []
        for filename in args.files:
            data.append(PerfStats(filename))
            file = open(filename, 'r')
            lines = file.readlines()[data[-1].result_header_idx + 1:]
            for line in lines:
                data[-1].append_line(line)

        # Plot the data.
        num_subplts = data[-1].num_subplts
        fig, axes = plt.subplots(num_subplts, len(data), constrained_layout=True)
        # Add a second y axis to each lower subplot.
        axes = axes.reshape(num_subplts, len(data))
        axes = np.vstack((axes, np.zeros([1, len(data)])))
        for i in range(len(data)):
            axes[num_subplts, i] = axes[num_subplts - 1, i].twinx()
        # Plot the data from each file.
        for i, d in enumerate(data):
            d.plot(axes[:, i])
        # Equalize all y axes.
        if args.equalize_axes:
            equalize_y_axes(axes)
        with warnings.catch_warnings():
            # Hide warnings due to the multiline title in PerfStats.plot()
            warnings.simplefilter('ignore', category=UserWarning)
        if args.plot_file:
            figure = plt.gcf()
            figure.set_size_inches(15, 20)
            plt.savefig(args.plot_file, dpi = 300, bbox_inches='tight')
        if args.show_plot:
            file_basename = os.path.basename(filename)
            if file_basename:
                plt.figure(1).suptitle(file_basename)
            plt.show()

    except KeyboardInterrupt:
        pass

