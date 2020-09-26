#ifndef TIMINGS_H
#define TIMINGS_H

#include <chrono>

#include "perfstats.h"

#if defined(SE_ENABLE_PERFSTATS) && SE_ENABLE_PERFSTATS

#define TICK(str) se::perfstats.sampleDurationStart(str);
#define TICKD(str) se::perfstats.sampleDurationStart(str, true);
#define TOCK(str) se::perfstats.sampleDurationEnd(str);

#else

#define TICK(str)
#define TICKD(str)
#define TOCK(str)

#endif

#endif // TIMINGS_H

