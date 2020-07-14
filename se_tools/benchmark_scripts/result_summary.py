#!/usr/bin/env python3.6

# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib
import numpy as np

from pylatex import *
import os

from evaluate_ate import *
from plot_stats import *



matplotlib.use('TkAgg')

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
            if line.split()[0] == 'frame':
                break
            param = line.split(':', 1)
            param_key = param[0]
            if param_key in param_key_list:
                param_value = param[1].lstrip().replace('\n', '')
                param_dict[param_key] = param_value

    return param_dict

if __name__ == '__main__':

    # Parse command line
    parser = argparse.ArgumentParser(description='''
    This script a test summary . 
    ''')
    parser.add_argument('--result-file', help='supereight result file [result.txt]')
    parser.add_argument('--summary-file', help='The file the summary is written to [summary]')
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

        # Read the result data
        result_data = (SEStats(args.result_file))
        for line in fileinput.input(args.result_file):
            result_data.append_line(line)

        # Write the computation time and memory stats
        with doc.create(Subsection('Stats')):
            # Computation time
            raws = "l l l l l l l l l"
            table_header = [" ", "Acq [s]", "Pre [s]", "Track [s]", "Int [s]", "Ray [s]", "Rend [s]", "Total [s]",
                            "RAM [MB]"]
            # ['BLANK', 'Acquisition', 'Preprocessing', 'Tracking', 'Integration', 'Raycasting', 'Rendering', 'Total']
            condition = [0, 1, 2, 3, 4, 5, 6, 7, 8]

            # Remove Tracking stats if ground truth is enabled -> Tracking is deactivated
            if enable_ground_truth:
                raws = raws[:-2]
                table_header.remove("Track [s]")
                condition.remove(3)
            # Remove Rendering stats if rendering is disabled -> Rendering is deactivated
            if not enable_render:
                raws = raws[:-2]
                table_header.remove("Rend [s]")
                condition.remove(6)
            # Remove Raycasting stats if ground truth is enabled and rendering disabled -> Raycasting is deactivated
            if enable_ground_truth and not enable_render:
                raws = raws[:-2]
                table_header.remove("Ray [s]")
                condition.remove(5)

            with doc.create(LongTable(raws)) as result_table:
                result_table.add_hline()
                result_table.add_row(table_header)

                # Skip the first frames as they drastically impact the min/max values
                start_frame = 4;
                acquisition_time   = result_data.acquisition_time[start_frame:]
                preprocessing_time = result_data.preprocessing_time[start_frame:]
                tracking_time      = result_data.tracking_time[start_frame:]
                integration_time   = result_data.integration_time[start_frame:]
                raycasting_time    = result_data.raycasting_time[start_frame:]
                rendering_time     = result_data.rendering_time[start_frame:]
                total_time         = result_data.total_time[start_frame:]

                result_mean   = [["Mean",
                                  '%.4f' % np.mean(acquisition_time),
                                  '%.4f' % np.mean(preprocessing_time),
                                  '%.4f' % np.mean(tracking_time),
                                  '%.4f' % np.mean(integration_time),
                                  '%.4f' % np.mean(raycasting_time),
                                  '%.4f' % np.mean(rendering_time),
                                  '%.4f' % np.mean(total_time),
                                  ''][i] for i in condition]
                result_median = [["Median",
                                  '%.4f' % np.median(acquisition_time),
                                  '%.4f' % np.median(preprocessing_time),
                                  '%.4f' % np.median(tracking_time),
                                  '%.4f' % np.median(integration_time),
                                  '%.4f' % np.median(raycasting_time),
                                  '%.4f' % np.median(rendering_time),
                                  '%.4f' % np.median(total_time),
                                  ''][i] for i in condition]
                result_std    = [["Std",
                                  '%.4f' % np.std(acquisition_time),
                                  '%.4f' % np.std(preprocessing_time),
                                  '%.4f' % np.std(tracking_time),
                                  '%.4f' % np.std(integration_time),
                                  '%.4f' % np.std(raycasting_time),
                                  '%.4f' % np.std(rendering_time),
                                  '%.4f' % np.std(total_time),
                                  ''][i] for i in condition]
                result_min    = [["Min",
                                  str('%.4f' % np.min(acquisition_time)) + " | " + str(np.where(result_data.acquisition_time == np.min(acquisition_time))[0][0]),
                                  str('%.4f' % np.min(preprocessing_time)) + " | " + str(np.where(result_data.preprocessing_time == np.min(preprocessing_time))[0][0]),
                                  str('%.4f' % np.min(tracking_time)) + " | " + str(np.where(result_data.tracking_time == np.min(tracking_time))[0][0]),
                                  str('%.4f' % np.min(integration_time)) + " | " + str(np.where(result_data.integration_time == np.min(integration_time))[0][0]),
                                  str('%.4f' % np.min(raycasting_time)) + " | " + str(np.where(result_data.raycasting_time == np.min(raycasting_time))[0][0]),
                                  str('%.4f' % np.min(rendering_time)) + " | " + str(np.where(result_data.rendering_time == np.min(rendering_time))[0][0]),
                                  str('%.4f' % np.min(total_time)) + " | " + str(np.where(result_data.total_time == np.min(total_time))[0][0]),
                                  ''][i] for i in condition]
                result_max    = [["Max",
                                  str('%.4f' % np.max(acquisition_time)) + " | " + str(np.where(result_data.acquisition_time == np.max(acquisition_time))[0][0]),
                                  str('%.4f' % np.max(preprocessing_time)) + " | " + str(np.where(result_data.preprocessing_time == np.max(preprocessing_time))[0][0]),
                                  str('%.4f' % np.max(tracking_time)) + " | " + str(np.where(result_data.tracking_time == np.max(tracking_time))[0][0]),
                                  str('%.4f' % np.max(integration_time)) + " | " + str(np.where(result_data.integration_time == np.max(integration_time))[0][0]),
                                  str('%.4f' % np.max(raycasting_time)) + " | " + str(np.where(result_data.raycasting_time == np.max(raycasting_time))[0][0]),
                                  str('%.4f' % np.max(rendering_time)) + " | " + str(np.where(result_data.rendering_time == np.max(rendering_time))[0][0]),
                                  str('%.4f' % np.max(total_time)) + " | " + str(np.where(result_data.total_time == np.max(total_time))[0][0]),
                                  '%.2f' % np.max(result_data.ram_usage)][i] for i in condition]

                result_table.add_hline()
                result_table.add_hline()
                result_table.add_row(result_mean)
                result_table.add_row(result_median)
                result_table.add_row(result_std)
                result_table.add_row(result_min)
                result_table.add_row(result_max)
                result_table.add_hline()

        # Create new page for plots
        doc.append(NewPage())

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

        axis.boxplot([[acquisition_time, preprocessing_time, tracking_time, integration_time,
                      raycasting_time, rendering_time, total_time][i] for i in condition])
        axis.set_xticklabels([['Acq', 'Pre', 'Track', 'Int',
                              'Ray', 'Rend', 'Total'][i] for i in condition])
        with doc.create(Figure(position='h!')) as time_plot:
            time_plot.add_plot(width=NoEscape( r'0.7\linewidth'))

        # Plot the result_data.
        fig, axes = plt.subplots(2, 1, constrained_layout=True)

        # Add a second y axis to each lower subplot.
        axes = axes.reshape(2, 1)
        axes = np.vstack((axes, np.zeros([1, 1])))
        axes[2, 0] = axes[1, 0].twinx()
        # Plot the result data from each file.
        result_data.plot(axes[:, 0])
        with warnings.catch_warnings():
            # Hide warnings due to the multiline title in SEStats.plot()
            warnings.simplefilter('ignore', category=UserWarning)

        with doc.create(Figure(position='h!')) as stats_plot:
            figure = plt.gcf()
            figure.set_size_inches(12, 9)
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
                            if os.exists(output_render_file):
                                with doc.create(SubFigure(
                                        position='b',
                                        width=NoEscape(r'0.5\linewidth'))) as volume_render:
                                    volume_render.add_image(output_render_file,
                                                            width=NoEscape(r'\linewidth'))
                                    volume_render.add_caption('Volume render ' + str(sub_frame))

    doc.generate_pdf(args.summary_file, clean_tex=False)


