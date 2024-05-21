#pragma once

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadEventWorkspace {
public:
  LoadEventWorkspace(const std::string &filename);
  std::vector<float> getLowValues();
  std::vector<float> getHighValues();
  double getProtonCharge();
  std::vector<float> getThetaValues();
  std::vector<float> getPhiValues();
  std::vector<std::array<double, 8>> getEvents();
  std::vector<int> getDetIDs();

private:
  HighFive::File m_file;
};