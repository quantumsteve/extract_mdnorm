#include "LoadExtrasWorkspace.h"

LoadExtrasWorkspace::LoadExtrasWorkspace(const std::string &filename) : m_file(filename, HighFive::File::ReadOnly) {}

Eigen::Matrix3f LoadExtrasWorkspace::getRotMatrix() const {
  HighFive::Group group = m_file.getGroup("expinfo_0");
  HighFive::DataSet dataset = group.getDataSet("goniometer_0");
  return dataset.read<Eigen::Matrix3f>();
}

std::vector<Eigen::Matrix3f> LoadExtrasWorkspace::getSymmMatrices() const {
  HighFive::Group group = m_file.getGroup("symmetryOps");
  auto n_elements = group.getNumberObjects();
  std::vector<Eigen::Matrix3f> symm;
  for (size_t i = 0; i < n_elements; ++i) {
    auto dataset = group.getDataSet("op_" + std::to_string(i));
    symm.push_back(dataset.read<Eigen::Matrix3f>());
  }
  return symm;
}

Eigen::Matrix3f LoadExtrasWorkspace::getUBMatrix() const {
  HighFive::DataSet dataset = m_file.getDataSet("ubmatrix");
  return dataset.read<Eigen::Matrix3f>();
}

std::vector<bool> LoadExtrasWorkspace::getSkipDets() const {
  std::vector<bool> skip_dets;
  HighFive::DataSet dataset = m_file.getDataSet("skip_dets");
  dataset.read(skip_dets);
  return skip_dets;
}
