#pragma once

#include <boost/histogram.hpp>
#include <boost/version.hpp>

namespace {
#if BOOST_VERSION >= 108000
using accumulator_type = boost::histogram::accumulators::count<double, true>;
#else
using accumulator_type = boost::histogram::accumulators::thread_safe<double>;
#endif

using histogram_type = boost::histogram::histogram<
    std::tuple<boost::histogram::axis::regular<float>, boost::histogram::axis::regular<float>,
               boost::histogram::axis::regular<float>>,
    boost::histogram::dense_storage<accumulator_type>>;
} // namespace
