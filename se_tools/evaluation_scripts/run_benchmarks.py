#!/usr/bin/python2
from _run import * 
from systemsettings import *
from datasets import *
import numpy as np 

TUM_RGBD_FR1 = [TUM_RGBD_FR1_XYZ, TUM_RGBD_FR1_DESK]
TUM_RGBD_FR2 = [TUM_RGBD_FR2_DESK]
TUM_RGBD_FR3 = [TUM_RGBD_FR3_LONG_OFFICE]
ICL = [ICL_NUIM_LIV_0, ICL_NUIM_LIV_1, ICL_NUIM_LIV_2, ICL_NUIM_LIV_3]

if __name__ == "__main__":
    results_dir = gen_results_dir(RESULTS_PATH)
    algorithm = KinectFusion(BIN_PATH)
    # -q --fps 10 --block-read 1

    # Find the best alignment between the gt and the computed trajectory.
    # It influences results a lot, we should really discuss this.
    algorithm.ate_align = True

    # All must be true for ICL-NUIM
    algorithm.ate_associate_identity = False  # 1to1 association gt-tra
    # When true the trajectory is offset by the first position.
    algorithm.ate_remove_offset = False
    algorithm.voxel_block = '8'
    algorithm.rendering_rate = '1'
    algorithm.bilateralFilter = False
    min_ate = 100.0
    run_results = {}

    # Open .log file for writing with line buffering
    with open(results_dir + '/resume.log', 'w', buffering=1) as f:
        # Write header.
        f.write('{:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<15} {:<}\n'\
                .format('dataset',
                'implementation',
                'ATE',
                'preprocessing',
                'integration',
                'raycasting',
                'computation',
                'noise_factor',
                'resolution'))
        run_counter = 0

        # Iterate over list of datasets to benchmark
        for sequence in ICL + TUM_RGBD_FR1 + TUM_RGBD_FR2 + TUM_RGBD_FR3:
            # Iterate over list of volume resolution values in voxels.
            for resol in [512]:
                # Iterate over map types.
                for version in ['sdf', 'ofusion']:
                    kernel_data = []
                    # occupancy maps need a lower noise threshold in general.
                    if version == 'ofusion':
                        mu = 0.01
                    else:
                        mu = 0.1
                    algorithm.impl = version
                    algorithm.volume_resolution = str(resol)
                    algorithm.compute_size_ratio = 2
                    algorithm.integration_rate = 1
                    algorithm.mu = mu
                    algorithm.init_pose = sequence.init_pose
                    algorithm.dump_volume = ".vtk"

                    # Run benchmark with current settings.
                    res = algorithm.run(sequence)

                    res['sequence'] = sequence.descr
                    res['noise_factor'] = mu
                    res['implementation'] = version
                    res['resolution'] = resol
                    run_results[run_counter] = res
                    run_counter += 1
                    kernel_data.append(res)
                    data = res['data']
                    f.write('{:<15} {:<15} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<}\n'
                            .format(res['sequence'], 
                            res['implementation'],
                            float(res['ate_mean']),
                            float(data['preprocessing']['mean']),
                            float(data['integration']['mean']),
                            float(data['raycasting']['mean']),
                            float(data['computation']['mean']),
                            float(res['noise_factor']),
                            int(res['resolution'])))

