#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib
import numpy as np
import os

from enum import Enum
from pylatex import *

from evaluate_ate import *
from plot_stats import *



matplotlib.use('TkAgg')

class Type(Enum):
    BOOL            = 0
    COORDINATES     = 1
    COUNT           = 2
    CURRENT         = 3
    DISTANCE        = 4
    DOUBLE          = 5
    DURATION        = 6
    ENERGY          = 7
    FRAME           = 8
    FREQUENCY       = 9
    INT             = 10
    ITERATION       = 11
    MEMORY          = 12
    ORIENTATION     = 13
    PERCENTAGE      = 14
    POSITION        = 15
    POWER           = 16
    TIME            = 17
    UNDEFINED       = 18
    VOLTAGE         = 19



def typeStringToType(type_string):
    switch={
        '[bool]'    : Type.BOOL,
        '[voxel]'   : Type.COORDINATES,
        '[count]'   : Type.COUNT,
        '[I]'       : Type.CURRENT,
        '[dm]'      : Type.DISTANCE, # d := \Delta for relative measurement
        '[double]'  : Type.DOUBLE,
        '[ds]'      : Type.DURATION, # d := \Delta for relative measurement
        '[J]'       : Type.ENERGY,
        '[1/s]'     : Type.FREQUENCY,
        '[int]'     : Type.INT,
        '[#]'       : Type.ITERATION,
        '[MB]'      : Type.MEMORY,
        '[-]'       : Type.ORIENTATION,
        '[%]'       : Type.PERCENTAGE,
        '[m]'       : Type.POSITION,
        '[W]'       : Type.POWER,
        '[s]'       : Type.TIME,
        '[und.]'    : Type.UNDEFINED,
        '[V]'       : Type.VOLTAGE,
    }
    return switch.get(type_string)



class Stats:
    def __init__(self, type):
        self.data = []
        self.type = type

    def stringToData(self, data_string):
        if data_string == '*':
            return None

        if self.type in [Type.BOOL, Type.COORDINATES, Type.COUNT, Type.INT, Type.ITERATION]:
            return int(data_string)

        return float(data_string)

    def summariseData(self):
        # mean, median, std, min, max, last
        start_iter = 4
        d = [i for i in self.data[start_iter:] if i] # Remove None values and data before start_iter
        if self.type in [Type.BOOL, Type.COUNT, Type.COORDINATES, Type.INT, Type.ITERATION]:
            mean_d   = '-'
            median_d = '-'
            std_d    = '-'
            min_d    = '-'
            max_d    = '-'
            last_d   = d[-1] if len(d) > 0 else '-'
            return [mean_d, median_d, std_d, min_d, max_d, last_d]

        if self.type in [Type.CURRENT, Type.DISTANCE, Type.DOUBLE, Type.ORIENTATION,
                         Type.PERCENTAGE, Type.POSITION, Type.TIME, Type.UNDEFINED]:
            mean_d   = '-'
            median_d = '-'
            std_d    = '-'
            min_d    = '-'
            max_d    = '-'
            last_d   = '%.4f' % d[-1] if len(d) > 0 else '-'
            return [mean_d, median_d, std_d, min_d, max_d, last_d]

        if self.type in [Type.MEMORY]:
            mean_d   = '-'
            median_d = '-'
            std_d    = '-'
            min_d    = '-'
            max_d    = '%.4f' % np.max(d)
            last_d   = '%.4f' % d[-1] if len(d) > 0 else '-'
            return [mean_d, median_d, std_d, min_d, max_d, last_d]

        if self.type in [Type.CURRENT, Type.DURATION, Type.ENERGY, Type.POWER, Type.VOLTAGE]:
            mean_d   = '%.4f' % np.mean(d)
            median_d = '%.4f' % np.median(d)
            std_d    = '%.4f' % np.std(d)
            min_d    = '%.4f' % np.min(d)
            max_d    = '%.4f' % np.max(d)
            last_d   = '%.4f' % d[-1] if len(d) > 0 else '-'
            return [mean_d, median_d, std_d, min_d, max_d, last_d]

        return ['-', '-', '-', '-', '-', '-']



param_key_list = [
    'Sequence name',
    'Voxel impl type',
    'Sensor type',
    'Enable ground truth',
    'Enable render',
    'Max frame',
    'Map size',
    'Map dim',
    'Map res',
    'Sensor intrinsics',
    'Sensor downsampling factor',
    'Near plane',
    'Far plane',
    'Rendering rate',
    'Output render file']



def read_parameter(result_file):
    file = open(result_file, 'r')
    lines = file.readlines()

    param_dict = {}
    for line in lines:
        if len(line.split()) > 0:
            if line == 'RESULT DATA\n':
                break
            param = line.split(':', 1)
            param_key = param[0]
            if param_key in param_key_list:
                param_value = param[1].lstrip().replace('\n', '')
                param_dict[param_key] = param_value

    return param_dict



def read_result(result_file):
    file = open(result_file, 'r')
    lines = file.readlines()

    result_dict = {}
    # Find line idx at which the results start (i.e. line after 'RESULT DATA\n')
    for idx, line in enumerate(lines):
        if line == 'RESULT DATA\n':
            result_header_idx = idx + 1
            break;

    result_header = lines[result_header_idx].split('\t')
    for idx, var in enumerate(result_header):
        var_name_unit_pair = var.split()
        if len(var_name_unit_pair) == 2:
            var_name = var_name_unit_pair[0]
            var_unit = var_name_unit_pair[1]
            var_type = typeStringToType(var_unit)
            result_dict[var_name] = Stats(var_type)

    result_data_lines =  lines[result_header_idx + 1:]
    for result_data_line in result_data_lines:
        result_iter_data = result_data_line.split('\t')
        for idx, var in enumerate(result_dict.values()):
            var_iter_data = var.stringToData(result_iter_data[idx])
            var.data += [var_iter_data];

    return result_dict

if __name__ == '__main__':

    # Parse command line
    parser = argparse.ArgumentParser(description='''This script a test summary .''')
    parser.add_argument('result_file', metavar='RESULT_IN', help='supereight result file [result.txt]')
    parser.add_argument('summary_file', metavar='PDF_OUT', help='The file the summary is written to [summary]')
    args = parser.parse_args()

    # Set document margins
    geometry_options = {"tmargin": "1cm", "lmargin": "1cm"}
    doc = Document(geometry_options=geometry_options)

    # Load parameter
    param_dict = read_parameter(args.result_file)
    enable_ground_truth     = param_dict["Enable ground truth"].lower() == 'true'
    enable_render           = param_dict["Enable render"].lower() == 'true'
    rendering_rate          = int(param_dict["Rendering rate"])
    output_render_file_base = param_dict["Output render file"] if "Output render file" in param_dict else None
    max_frame               = int(param_dict["Max frame"])

    # Compute ATE data if ground truth is disabled -> Tracking activated
    if not enable_ground_truth:
        evaluation = EvaluateATE(args.result_file)
        evaluation.associate()
        evaluation.calculate()

    # Write case parameters
    with doc.create(Section('Result Summary')):
        with doc.create(Subsection('System Parameter')):
            with doc.create(LongTable("l l")) as param_table:
                param_table.add_hline()
                param_table.add_row(["Pipeline", ""])
                param_table.add_hline()
                param_table.add_hline()
                param_table.add_row("Voxel impl type",       param_dict["Voxel impl type"])
                param_table.add_row("Sensor type",           param_dict["Sensor type"])
                param_table.add_row("Sequence name",         param_dict["Sequence name"])
                param_table.add_row("Enable ground truth",   param_dict["Enable ground truth"])
                param_table.add_row("Enable render",         param_dict["Enable render"])
                param_table.add_row("Max frame",             param_dict["Max frame"])
                param_table.add_empty_row()
                param_table.add_hline()

                param_table.add_row(["Map", ""])
                param_table.add_hline()
                param_table.add_hline()
                param_table.add_row("Dim",        param_dict["Map dim"])
                param_table.add_row("Size",       param_dict["Map size"])
                param_table.add_row("Resolution", param_dict["Map res"])
                param_table.add_empty_row()
                param_table.add_hline()

                param_table.add_row(["Sensor", ""])
                param_table.add_hline()
                param_table.add_hline()
                param_table.add_row("Intrinsics",         param_dict["Sensor intrinsics"])
                param_table.add_row("Down-sample factor", param_dict["Sensor downsampling factor"])
                param_table.add_row("Near plane",         param_dict["Near plane"])
                param_table.add_row("Far plane",          param_dict["Far plane"])
                param_table.add_hline()

        # TODO:
        # with doc.create(Subsection('System Description')):
        #     doc.append('Some regular text and some')


        # Create new page for stats
        doc.append(NewPage())

        result_dict = read_result(args.result_file)

        # Write the computation time and memory stats
        with doc.create(Subsection('Stats')):
            # Computation time
            raws = "l l l l l l l"
            table_header = [" ", "mean", "median", "std", "min", "max", "last"]

            with doc.create(LongTable(raws)) as result_table:
                result_table.add_hline()
                result_table.add_row(table_header)
                result_table.add_hline()
                result_table.add_hline()
                for var_name, var_data in result_dict.items():
                    var_data_summary = var_data.summariseData()
                    var_table_line = [var_name] + var_data_summary
                    result_table.add_row(var_table_line)
                    result_table.add_hline()

        # Create new page for plots
        doc.append(NewPage())

        result_data = PerfStats(args.result_file)
        result_file = fileinput.input(args.result_file)
        for line in result_file:
            if result_file.lineno() > result_data.result_header_idx + 1:
                result_data.append_line(line)

        fig, axis = plt.subplots(1, 1, constrained_layout=True)
        axis.set_title('Computation time box plot')

        # ['Acquisition', 'Preprocessing', 'Tracking', 'Integration', 'Raycasting', 'Rendering', 'Total']
        condition = [0, 1, 2, 3, 4, 5, 6]

        # Remove Tracking stats if ground truth is enabled -> Tracking is deactivated
        if enable_ground_truth:
            condition.remove(2)
        # Remove Rendering stats if rendering is disabled -> Rendering is deactivated
        if not enable_render:
            condition.remove(5)
        # Remove Raycasting stats if ground truth is enabled and rendering disabled -> Raycasting is deactivated
        if enable_ground_truth and not enable_render:
            condition.remove(4)

        start_frame = 4;
        acquisition_time = result_data.acquisition_time[start_frame:]
        preprocessing_time = result_data.preprocessing_time[start_frame:]
        tracking_time = result_data.tracking_time[start_frame:]
        integration_time = result_data.integration_time[start_frame:]
        raycasting_time = result_data.raycasting_time[start_frame:]
        rendering_time = result_data.rendering_time[start_frame:]
        total_time = result_data.total_time[start_frame:]

        axis.boxplot([[acquisition_time, preprocessing_time, tracking_time, integration_time,
                      raycasting_time, rendering_time, total_time][i] for i in condition])
        axis.set_xticklabels([['Acq', 'Pre', 'Track', 'Int',
                              'Ray', 'Rend', 'Total'][i] for i in condition])
        with doc.create(Figure(position='h!')) as time_plot:
            time_plot.add_plot(width=NoEscape( r'0.7\linewidth'))

        doc.append(NewPage())

        # Plot the result_data.
        num_subplts = result_data.num_subplts
        fig, axes = plt.subplots(num_subplts, 1, constrained_layout=True)
        # Add a second y axis to each lower subplot.
        axes = axes.reshape(num_subplts, 1)
        axes = np.vstack((axes, 1))
        axes[num_subplts, 0] = axes[num_subplts - 1, 0].twinx()
        # Plot the result data from each file.
        result_data.plot(axes[:, 0])
        with warnings.catch_warnings():
            # Hide warnings due to the multiline title in SEStats.plot()
            warnings.simplefilter('ignore', category=UserWarning)

        with doc.create(Figure(position='h!')) as stats_plot:
            figure = plt.gcf()
            figure.set_size_inches(15, 20)
            stats_plot.add_plot(width=NoEscape(r'\linewidth'), dpi = 300)

        # Write ATE data if ground truth is disabled -> Tracking activated
        if not enable_ground_truth:
            with doc.create(Subsection('ATE Error')):
                # Generate ATE table
                with doc.create(LongTable("l l l l l l")) as ate_table:
                    ate_table.add_hline()
                    ate_table.add_row(["ATE RMSE", "ATE mean", "ATE median", "ATE std", "ATE min", "ATE max"])
                    ate_table.add_hline()
                    ate_table.add_hline()
                    ate_data = evaluation.get_ate()
                    ate_data_round = ['%.4f' % elem for elem in ate_data]
                    ate_table.add_row(ate_data_round)
                    ate_table.add_hline()

                fig, axis = plt.subplots(1, 1, constrained_layout=True)
                axis.set_title('ATE box plot')
                trans_error = evaluation.get_trans_error()
                axis.boxplot(trans_error)
                with doc.create(Figure(position='h!')) as box_plot:
                    box_plot.add_plot(width=NoEscape(r'0.5\linewidth'))

        doc.append(NewPage())

        if enable_render and output_render_file_base:
            with doc.create(Subsection('Render')):
                if rendering_rate < 50:
                    saving_rate = floor(50 / rendering_rate) * rendering_rate
                else:
                    saving_rate = rendering_rate

                for frame in range(saving_rate, max_frame, 2 * saving_rate):
                    with doc.create(Figure(position='h!')) as volume_renders:
                        for sub_frame in [frame, min(frame + saving_rate, max_frame)]:
                            file_ending = "_frame_" + str(sub_frame).zfill(4) + ".png"
                            output_render_file = output_render_file_base + file_ending
                            if os.path.exists(output_render_file):
                                with doc.create(SubFigure(
                                        position='b',
                                        width=NoEscape(r'0.5\linewidth'))) as volume_render:
                                    volume_render.add_image(output_render_file,
                                                            width=NoEscape(r'\linewidth'))
                                    volume_render.add_caption('Volume render ' + str(sub_frame))

    doc.generate_pdf(args.summary_file, clean_tex=False)


