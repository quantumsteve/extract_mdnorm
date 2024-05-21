#pragma once

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadEventWorkspace {
public:
  LoadEventWorkspace(const std::string &filename);
  std::vector<float> getLowValues() const;
  std::vector<float> getHighValues() const;
  double getProtonCharge() const;
  std::vector<float> getThetaValues() const;
  std::vector<float> getPhiValues() const;
  std::vector<std::array<double, 8>> getEvents() const;
  std::vector<int> getDetIDs() const;

private:
  HighFive::File m_file;
};
