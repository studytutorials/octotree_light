#ifndef __STR_UTILS_H
#define __STR_UTILS_H

#include <sstream>
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
static std::vector<std::string> split_string(
    const std::string& s,
    const char         delim,
    const bool         ignore_consec = false) {

  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    // Empty items result from consecutive occurences of the delimiter.
    if (!ignore_consec || (ignore_consec && (item.size() > 0))) {
      elems.push_back(item);
    }
  }
  return elems;
}

#endif

