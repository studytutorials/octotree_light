#ifndef __STR_UTILS_HPP
#define __STR_UTILS_HPP

#include <string>
#include <vector>

/**
 * Split a string into a vector of substrings based on a delimiter.
 *
 * \param[in] s The string to split.
 * \param[in] delim The delimiter to use.
 * \param[in] ignore_consec Treat consecutive delimiters as a single delimiter.
 * \return The vector containing the substrings.
 */
std::vector<std::string> split_string(
    const std::string& s,
    const char         delim,
    const bool         ignore_consec = false);

#endif

