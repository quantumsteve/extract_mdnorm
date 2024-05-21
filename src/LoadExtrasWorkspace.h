#pragma once

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadExtrasWorkspace {
public:
  LoadExtrasWorkspace(const std::string &filename);
  Eigen::Matrix3f getRotMatrix() const;
  std::vector<Eigen::Matrix3f> getSymmMatrices() const;
  Eigen::Matrix3f getUBMatrix() const;
  std::vector<bool> getSkipDets() const;

private:
  HighFive::File m_file;
};
