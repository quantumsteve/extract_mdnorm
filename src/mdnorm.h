#pragma once

#include "parameters.h"

#include <boost/histogram.hpp>

namespace{
using histogram_type = boost::histogram::histogram<
    std::tuple<boost::histogram::axis::regular<float>, boost::histogram::axis::regular<float>,
               boost::histogram::axis::regular<float>>,
    boost::histogram::dense_storage<boost::histogram::accumulators::thread_safe<double>>>;
}

void mdnorm(parameters &params, histogram_type &signal, histogram_type& h);