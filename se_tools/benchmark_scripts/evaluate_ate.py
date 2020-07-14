#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

#
# FYI - Modified to add functionality. 
#

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

import sys
import numpy
import argparse

class EvaluateATE:

    def __init__(self, result_file):
        self.ground_truth_file = None
        self.result_file       = result_file
        self.result_idx        = None

    def __str__(self):
        return "".join(["========   ATE RESULTS   =======",
                        "ATE RMSE:    {:f} m".format(self.ate_rmse()),
                        "ATE mean:    {:f} m".format(numpy.mean(self.trans_error)),
                        "ATE median:  {:f} m".format(numpy.median(self.trans_error)),
                        "ATE std:     {:f} m".format(numpy.std(self.trans_error)),
                        "ATE min:     {:f} m".format(numpy.min(self.trans_error)),
                        "ATE max:     {:f} m".format(numpy.max(self.trans_error))])

    def read_result_file(self, result_file):
        file = open(result_file, 'r')
        lines = file.readlines()
        # Find line idx at which the results start (i.e. line after 'frame')
        for idx, line in enumerate(lines):
            if len(line.split()) > 0:
                if str(line.split()[0]) == 'Ground':
                    self.ground_truth_file = str(line.split()[3])
                elif str(line.split()[0]) == 'frame':
                    self.result_idx = idx + 1
                    break
        if self.ground_truth_file == "":
            raise ValueError('No ground truth file provided')
        result = [line.split() for line in lines[self.result_idx:]]
        #                     frame    | X           | Y           | Z
        result_trajectory = [[int(r[0]), float(r[10]), float(r[11]), float(r[12])] for r in result]
        return result_trajectory

    def read_ground_truth_file(self, ground_truth_file):
        file = open(ground_truth_file, 'r')
        lines = file.readlines()
        ground_truth = [line.split() for line in lines if len(line)>0 and line[0]!="#"]
        line_length = len(ground_truth[0])
        #                           X                        | Y                        | Z
        ground_truth_trajectory = [[float(g[line_length - 7]), float(g[line_length - 6]), float(g[line_length - 5])] for g in ground_truth]
        return ground_truth_trajectory

    def associate(self):
        self.result_trajectory       = self.read_result_file(self.result_file)
        self.ground_truth_trajectory = self.read_ground_truth_file(self.ground_truth_file)
        self.matches = [[p[1:], self.ground_truth_trajectory[p[0]]] for p in self.result_trajectory]

    #
    # Direct pairwise ATE evaluation.
    #
    def calculate(self):
        self.trans_error = []

        for res_p, gt_p in self.matches:
            err = (abs(res_p[0] - gt_p[0]), abs(res_p[1] - gt_p[1]), abs(res_p[2] - gt_p[2]))
            err_2 = (err[0] * err[0], err[1] * err[1], err[2] * err[2])
            self.trans_error.append(numpy.sqrt(sum(err_2)))

    def get_trans_error(self):
        return self.trans_error

    def ate_rmse(self):
        return numpy.sqrt(numpy.dot(self.trans_error,self.trans_error) / len(self.trans_error))

    def get_ate(self):
        ate_data = [self.ate_rmse(),
                numpy.mean(self.trans_error),
                numpy.median(self.trans_error),
                numpy.std(self.trans_error),
                numpy.min(self.trans_error),
                numpy.max(self.trans_error)]
        return  ate_data

    def print_ate(self):
        print(str(self))

    def save_ate(self):
        f = open(self.result_file, "r")
        contents = f.readlines()
        contents.insert(self.result_idx - 2, str(self))
        contents = "".join(contents)

        f = open(self.result_file, "w")
        f.write(contents)
        f.close()


if __name__=="__main__":
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('--result-file', help='estimated trajectory (supereight result file)')
    parser.add_argument('--save-ate',   help='save ATE values to result file', action="store_true")
    parser.add_argument('--print-ate',  help='print ATE values to terminal', action="store_true")
    args = parser.parse_args()

    evaluation = EvaluateATE(args.result_file)
    evaluation.associate()
    evaluation.calculate()
    if args.print_ate:
        evaluation.print_ate()
    if args.save_ate:
        evaluation.save_ate()

