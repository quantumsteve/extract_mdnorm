#pragma once

#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <string>
#include <unordered_map>
#include <vector>

class LoadExtrasWorkspace {
public:
  LoadExtrasWorkspace(const std::string &filename);
  Eigen::Matrix3f getRotMatrix();
  std::vector<Eigen::Matrix3f> getSymmMatrices();
  Eigen::Matrix3f getUBMatrix();
  std::vector<bool> getSkipDets();

private:
  HighFive::File m_file;
};