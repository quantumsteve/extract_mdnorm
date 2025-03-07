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
  std::vector<std::array<double, 8>> getEvents() const {
    std::vector<std::array<double, 8>> events;
    this->updateEvents(events);
    return events;
  }
  void updateEvents(std::vector<std::array<float, 8>> &events) const;
  void updateEvents(std::vector<std::array<double, 8>> &events) const;
  void updateEvents(Eigen::Matrix<float, Eigen::Dynamic, 8> &events) const;
  void updateEvents(Eigen::Matrix<float, 3, Eigen::Dynamic> &events) const;

  std::vector<int> getDetIDs() const;

  void updateBoxType(std::vector<int> &boxType) const;
  void updateExtents(Eigen::Matrix<double, Eigen::Dynamic, 6> &extents) const;
  void updateSignal(Eigen::Matrix<double, Eigen::Dynamic, 2> &signal) const;
  void updateEventIndex(Eigen::Matrix<uint64_t, Eigen::Dynamic, 2> &eventIndex) const;

private:
  HighFive::File m_file;
};
