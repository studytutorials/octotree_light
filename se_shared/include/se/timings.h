#ifndef TIMINGS_H
#define TIMINGS_H

#include <chrono>

#include "perfstats.h"

#if defined(SE_ENABLE_TICKTOCK) && SE_ENABLE_TICKTOCK

#define TICK() { \
  const auto tickdata = std::chrono::steady_clock::now();

#define TOCK(str, size) \
  const auto tockdata = std::chrono::steady_clock::now(); \
  const auto diff = tockdata - tickdata; \
  stats.sample(str, std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count()); \
}

#else

#define TICK()
#define TOCK(str, size)

#endif

#endif // TIMINGS_H

