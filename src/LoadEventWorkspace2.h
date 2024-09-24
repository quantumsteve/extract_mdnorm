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
private:
  hid_t m_file;
};

