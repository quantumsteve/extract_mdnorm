#pragma once

#include <Eigen/Core>

#include <array>
#include <atomic>
#include <vector>

class MDNorm {
public:
  MDNorm(const std::vector<double> &hX, const std::vector<double> &kX, const std::vector<double> &lX);
  /**
   * Calculate the points of intersection for the given detector with cuboid
   * surrounding the detector position in HKL
   * @param intersections A list of intersections in HKL space
   * @param theta Polar angle withd detector
   * @param phi Azimuthal angle with detector
   * @param transform Matrix to convert frm Q_lab to HKL (2Pi*R *UB*W*SO)^{-1}
   * @param lowvalue The lowest momentum or energy transfer for the trajectory
   * @param highvalue The highest momentum or energy transfer for the trajectory
   */
  void calculateIntersections(std::vector<std::array<double, 4>> &intersections, double theta,
                              double phi, const Eigen::Matrix3d &transform, double lowvalue,
                              double highvalue);
  /**
   * Calculate the normalization among intersections on a single detector
   * in 1 specific SpectrumInfo/ExperimentInfo
   * @param intersections: intersections
   * @param solid: proton charge
   * @param yValues: diffraction intersection integral and common to sample and background
   * @param vmdDims: MD dimensions
   * @param pos: position from intersecton for memory efficiency
   * @param posNew: transformed positions
   * @param signalArray: (output) normalization
   * @param solidBkgd: background proton charge
   * @param bkgdSignalArray: (output) background normalization
   */
  void calcSingleDetectorNorm(const std::vector<std::array<double, 4>> &intersections, const double &solid,
                              std::vector<double> &yValues, const size_t &vmdDims, std::vector<float> &pos,
                              std::vector<float> &posNew, std::vector<std::atomic<double>> &signalArray);

private:
  size_t getLinearIndexAtCoord(const float *coords);
  float m_origin[3] = {-10., -10, -0.1};
  float m_boxLength[3] = {0.1, 0.1, 0.2};
  size_t m_indexMax[3] = {200, 200, 1};
  size_t m_indexMaker[3];
  std::vector<double> m_hX, m_kX, m_lX, m_eX;
};
