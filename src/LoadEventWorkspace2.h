#pragma once

#include "hdf5.h"

#include "Eigen/Dense"

#include <string>

class LoadEventWorkspace2 {
public:
  LoadEventWorkspace2(const std::string &filename);
  ~LoadEventWorkspace2();
  double getProtonCharge() const;
  void updateEvents(Eigen::Matrix<float, Eigen::Dynamic, 3> &events) const;

  void updateBoxType(std::vector<unsigned char> &boxType) const;
  void updateExtents(Eigen::Matrix<float, Eigen::Dynamic, 6> &extents) const;
  void updateSignal(std::vector<float> &signal) const;
  void updateEventIndex(Eigen::Matrix<uint64_t, Eigen::Dynamic, 2> &eventIndex) const;

private:
  hid_t m_file;
};

