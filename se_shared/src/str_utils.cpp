#include "se/str_utils.hpp"

#include <sstream>

std::vector<std::string> split_string(
    const std::string& s,
    const char         delim,
    const bool         ignore_consec) {

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

