#include "LoadSolidAngleWorkspace.h"

LoadSolidAngleWorkspace::LoadSolidAngleWorkspace(const std::string &filename)
    : m_file(filename, HighFive::File::ReadOnly) {}

std::vector<std::vector<double>> LoadSolidAngleWorkspace::getSolidAngleValues() const {
  std::vector<std::vector<double>> solidAngleWS;
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("workspace");
  HighFive::DataSet dataset = group2.getDataSet("values");
  std::vector<size_t> dims = dataset.getDimensions();
  // std::vector<double> read_data;
  dataset.read(solidAngleWS);
  // for (const double value : read_data)
  //  solidAngleWS.push_back({value});
  return solidAngleWS;
}

std::unordered_map<int32_t, size_t> LoadSolidAngleWorkspace::getSolidAngDetToIdx() const {
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("instrument");
  HighFive::Group group3 = group2.getGroup("detector");
  HighFive::DataSet dataset = group3.getDataSet("detector_count");
  std::vector<size_t> dims = dataset.getDimensions();
  std::vector<int> dc_data;
  dataset.read(dc_data);

  dataset = group3.getDataSet("detector_list");
  dims = dataset.getDimensions();
  std::vector<int> detIDs;
  dataset.read(detIDs);

  std::unordered_map<int32_t, size_t> solidAngDetToIdx;
  int detector{0};
  size_t idx{0};
  for (auto &value : dc_data) {
    for (int i = 0; i < value; ++i) {
      solidAngDetToIdx.emplace(detIDs[detector++], idx);
    }
    ++idx;
  }
  return solidAngDetToIdx;
}
