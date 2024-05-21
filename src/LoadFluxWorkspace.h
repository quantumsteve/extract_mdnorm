#pragma once

#include <boost/histogram.hpp>

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadFluxWorkspace {
public:
  using reg = boost::histogram::axis::regular<float>;
  LoadFluxWorkspace(const std::string &filename);
  reg getFluxAxis() const;
  std::vector<std::vector<double>> getFluxValues() const;
  std::unordered_map<int32_t, size_t> getFluxDetToIdx() const;
  size_t getNDets() const;

private:
  HighFive::File m_file;
};
