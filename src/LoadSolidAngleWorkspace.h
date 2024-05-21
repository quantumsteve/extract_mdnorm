#pragma once

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadSolidAngleWorkspace {
public:
  LoadSolidAngleWorkspace(const std::string &filename);
  std::vector<std::vector<double>> getSolidAngleValues() const;
  std::unordered_map<int32_t, size_t> getSolidAngDetToIdx() const;

private:
  HighFive::File m_file;
};
