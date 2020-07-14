# SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2020 Nils Funk
# SPDX-License-Identifier: BSD-3-Clause

import sys
import time
class ProgressBar:
    def __init__(self, bar_length, total_steps, unit, lines_subplot):
        self.bar_length     = bar_length
        self.total_steps    = total_steps
        self.unit           = unit
        self.lines_subplot  = lines_subplot

    def _up(self, step=1):
        # My terminal breaks if we don't flush after the escape-code
        for i in range(step):
            sys.stdout.write('\x1b[1A')
        sys.stdout.flush()

    def _down(self, step=1):
        # I could use '\x1b[1B' here, but newline is faster and easier
        for i in range(step):
            sys.stdout.write('\n')
        sys.stdout.flush()

    def _clear_line(self):
        sys.stdout.write("\033[K")

    def _print_status(self, step, summary = ""):
        self._clear_line()
        print("Evaluating %s %d of %d" % (self.unit, step, self.total_steps))
        self._clear_line()
        print(summary)

    def _print_bar(self, percent):
        arrow   = '*' * int(percent / 100 * self.bar_length)
        spaces  = ' ' * (self.bar_length - len(arrow))
        print('%3d %% [%s%s]' % (percent, arrow, spaces), end='\r')

    def start(self):
        curr_step   = 0
        percent     = 0
        self._print_status(curr_step, "Start benchmark")
        self._print_bar(percent)

    def end(self):
        curr_step    = self.total_steps
        percent = 100
        self._print_status(curr_step, "Finished benchmark")
        self._print_bar(percent)
        for i in range(1 + self.lines_subplot):
            sys.stdout.write('\n')
        sys.stdout.flush()

    def update(self, curr_step, summary):
        percent = int(100 * (curr_step - 1) / self.total_steps)
        self._print_status(curr_step, summary)
        self._print_bar(percent)

    def jump_main_to_sub_plot(self):
        self._down(1)

    def jump_sub_to_main_plot(self):
        self._up((1 + self.lines_subplot))

    def flash(self):
        sys.stdout.write('\x1b[1A\x1b[1A')
        sys.stdout.flush()
        self._clear_line()




