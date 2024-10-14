#pragma once

#include <boost/histogram.hpp>
#include <Eigen/Core>

#include <tuple>

struct parameters
{
  using reg = boost::histogram::axis::regular<float>;
  std::tuple<reg, reg, reg> axes;
  Eigen::Matrix3f W;
  std::string solidAngleFilename;
  std::string fluxFilename;
  std::string eventPrefix;
  int eventMin;
  int eventMax;
  int eventStart;
  int eventStop;
};
