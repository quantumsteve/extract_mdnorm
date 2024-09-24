#include "LoadEventWorkspace2.h"

#include <iostream>

LoadEventWorkspace2::LoadEventWorkspace2(const std::string &filename) {
  m_file = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
}

LoadEventWorkspace2::~LoadEventWorkspace2() {
  H5Fclose(m_file);
}

double LoadEventWorkspace2::getProtonCharge() const {
  const char *DATASET = "MDEventWorkspace/experiment0/logs/gd_prtn_chrg/value";	
  hid_t dset = H5Dopen(m_file, DATASET, H5P_DEFAULT);
  double protonCharge; 
  auto status = H5Dread (dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &protonCharge);
  status = H5Dclose (dset);
  return protonCharge;
}

void LoadEventWorkspace2::updateEvents(Eigen::Matrix<float, Eigen::Dynamic, 3> &events) const {
  const char *DATASET = "MDEventWorkspace/event_data/position";
  hid_t dataset = H5Dopen(m_file, DATASET, H5P_DEFAULT);

  hid_t dataspace = H5Dget_space(dataset);
  hsize_t rank = H5Sget_simple_extent_ndims(dataspace);
  std::array<hsize_t,2> dims_out;
  auto status = H5Sget_simple_extent_dims(dataspace, dims_out.data(), NULL);

  events.resize(dims_out[1], dims_out[0]);
  status = H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, events.data());

  status = H5Dclose(dataset);
  status = H5Sclose(dataspace);
}

