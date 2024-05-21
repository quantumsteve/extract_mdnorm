#include "LoadEventWorkspace.h"

#include <boost/math/constants/constants.hpp>

LoadEventWorkspace::LoadEventWorkspace(const std::string &filename) : m_file(filename, HighFive::File::ReadOnly) {}

std::vector<float> LoadEventWorkspace::getLowValues() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("logs");
  HighFive::Group group4 = group3.getGroup("MDNorm_low");
  auto dataset = group4.getDataSet("value");
  std::vector<float> lowValues;
  dataset.read(lowValues);
  return lowValues;
}

std::vector<float> LoadEventWorkspace::getHighValues() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("logs");
  HighFive::Group group4 = group3.getGroup("MDNorm_high");
  auto dataset = group4.getDataSet("value");
  std::vector<float> highValues;
  dataset.read(highValues);
  return highValues;
}

double LoadEventWorkspace::getProtonCharge() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("logs");
  HighFive::Group group4 = group3.getGroup("gd_prtn_chrg");
  auto dataset = group4.getDataSet("value");
  double protonCharge;
  dataset.read(protonCharge);
  return protonCharge;
}

std::vector<float> LoadEventWorkspace::getThetaValues() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("instrument");
  HighFive::Group group4 = group3.getGroup("physical_detectors");
  auto dataset = group4.getDataSet("polar_angle");
  std::vector<float> thetaValues;
  dataset.read(thetaValues);

#pragma omp parallel for
  for (auto &value : thetaValues)
    value *= boost::math::float_constants::degree;
  return thetaValues;
}

std::vector<float> LoadEventWorkspace::getPhiValues() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("instrument");
  HighFive::Group group4 = group3.getGroup("physical_detectors");
  auto dataset = group4.getDataSet("azimuthal_angle");
  std::vector<float> phiValues;
  dataset.read(phiValues);

#pragma omp parallel for
  for (auto &value : phiValues)
    value *= boost::math::float_constants::degree;
  return phiValues;
}

std::vector<int> LoadEventWorkspace::getDetIDs() {
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("experiment0");
  HighFive::Group group3 = group2.getGroup("instrument");
  HighFive::Group group4 = group3.getGroup("physical_detectors");
  auto dataset = group4.getDataSet("detector_number");
  std::vector<int> detIDs;
  dataset.read(detIDs);
  return detIDs;
}

std::vector<std::array<double, 8>> LoadEventWorkspace::getEvents() {
  // const char *EventHeaders[] = {"signal, errorSquared, center (each dim.)",
  //                              "signal, errorSquared, expInfoIndex, goniometerIndex, detectorId, center (each "
  //                              "dim.)"};
  // https://github.com/mantidproject/mantid/blob/c3ea43e4605f6898b84bd95c1196ccd8035364b1/Framework/DataObjects/src/BoxControllerNeXusIO.cpp#L27
  HighFive::Group group = m_file.getGroup("MDEventWorkspace");
  HighFive::Group group2 = group.getGroup("event_data");
  auto dataset = group2.getDataSet("event_data");
  std::vector<std::array<double, 8>> events;
  dataset.read(events);
  return events;
}
