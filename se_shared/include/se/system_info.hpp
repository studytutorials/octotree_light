// SPDX-FileCopyrightText: 2020 Smart Robotics Lab, Imperial College London
// SPDX-FileCopyrightText: 2020 Sotiris Papatheodorou
// SPDX-License-Identifier: BSD-3-Clause

/** \file */

#ifndef __SYSTEM_INFO_HPP
#define __SYSTEM_INFO_HPP

#include <cstdio>
#include <cstring>



namespace se {

  /** Return the RAM usage in bytes of the calling program.
   *
   * This function depends on the underlying OS. It's implemented for the
   * following:
   * - Linux: It parses /proc/self/status for the RAM usage (VmRSS).
   * - Other: Always returns 0.
   *
   * \return The RAM usage in bytes.
   */
  static inline size_t ram_usage_self() {
#ifdef __linux__
    // Open the status file in the /proc pseudo-filesystem for the current
    // process
    FILE* file = fopen("/proc/self/status", "r");
    if (file == NULL) {
      return 0;
    }

    // Read the file line-by-line
    char line[512];
    while (fgets(line, sizeof(line), file) != NULL) {
      // Find the line containing the virtual memory resident set size
      if (strncmp(line, "VmRSS", 5) == 0) {
        // Find the substring that contains the number
        char *number_str = strpbrk(line, "0123456789");
        const size_t number_str_len = strcspn(number_str, " ");
        // Find the substring that contains the unit
        char *unit_str = strpbrk(number_str, "BkMG");
        const size_t unit_str_len = strcspn(unit_str, "\n");
        // Null-terminate the substrings (destroying the line read in the
        // process)
        number_str[number_str_len] = '\0';
        unit_str[unit_str_len] = '\0';
        // Get the number and scale according to the unit
        const size_t number = atoll(number_str);
        size_t scale;
        switch (unit_str[0]) {
          case 'B': // B
            scale = 1;
            break;
          case 'k': // kB
            scale = 1024;
            break;
          case 'M': // MB
            scale = 1024 * 1024;
            break;
          case 'G': // GB
            scale = 1024 * 1024 * 1024;
            break;
          default: // Assume kB (seems to be the most common unit in /proc)
            scale = 1024;
            break;
        }
        fclose(file);
        return number * scale;
      } else {
        continue;
      }
    }

    fclose(file);
    // The VmRSS line was not found
#endif
    return 0;
  }

} // namespace se

#endif

