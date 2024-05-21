#include "LoadFluxWorkspace.h"

LoadFluxWorkspace::LoadFluxWorkspace(const std::string &filename) : m_file(filename, HighFive::File::ReadOnly) {}

LoadFluxWorkspace::reg LoadFluxWorkspace::getFluxAxis() {
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("workspace");
  HighFive::DataSet dataset = group2.getDataSet("axis1");
  auto dims = dataset.getDimensions();
  std::vector<float> read_data_f;
  dataset.read(read_data_f);
  return reg(read_data_f.size() - 1, read_data_f.front(), read_data_f.back(), "integrFlux_x");
}

std::vector<std::vector<double>> LoadFluxWorkspace::getFluxValues() {
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("workspace");
  auto dataset = group2.getDataSet("values");
  auto dims = dataset.getDimensions();
  std::vector<double> read_data;
  dataset.read(read_data);

  std::vector<std::vector<double>> integrFlux_y{1};
  for (size_t j = 0; j < dims[1]; ++j)
    integrFlux_y[0].push_back(read_data[j]);
  return integrFlux_y;
}

std::unordered_map<int32_t, size_t> LoadFluxWorkspace::getFluxDetToIdx() {
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("instrument");
  HighFive::Group group3 = group2.getGroup("detector");
  auto dataset = group3.getDataSet("detector_count");
  auto dims = dataset.getDimensions();
  std::vector<int> dc_data;
  dataset.read(dc_data);

  dataset = group3.getDataSet("detector_list");
  dims = dataset.getDimensions();
  std::vector<int> detIDs;
  dataset.read(detIDs);

  std::unordered_map<int32_t, size_t> fluxDetToIdx;
  int detector = 0;
  int idx = 0;
  for (auto &value : dc_data) {
    for (int i = 0; i < value; ++i) {
      fluxDetToIdx.emplace(detIDs[detector++], idx);
    }
    ++idx;
  }
  return fluxDetToIdx;
}

size_t LoadFluxWorkspace::getNDets() {
  HighFive::Group group = m_file.getGroup("mantid_workspace_1");
  HighFive::Group group2 = group.getGroup("instrument");
  auto group3 = group2.getGroup("physical_detectors");
  auto dataset = group3.getDataSet("number_of_detectors");
  size_t ndets;
  dataset.read(ndets);
  return ndets;
}
