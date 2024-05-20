#pragma once

#include <Eigen/Core>
#include <boost/histogram.hpp>

#include <array>
#include <vector>

class MDNorm {
public:
  using histogram_type = boost::histogram::histogram<
      std::tuple<boost::histogram::axis::regular<float>, boost::histogram::axis::regular<float>,
                 boost::histogram::axis::regular<float>>,
      boost::histogram::dense_storage<boost::histogram::accumulators::thread_safe<double>>>;

  MDNorm(const std::vector<float> &hX, const std::vector<float> &kX, const std::vector<float> &lX);
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
  void calculateIntersections(std::vector<std::array<float, 4>> &intersections, float theta, float phi,
                              const Eigen::Matrix3f &transform, float lowvalue, float highvalue);
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
  void calculateIntersections(histogram_type &h, std::vector<std::array<float, 4>> &intersections, const float theta,
                              const float phi, const Eigen::Matrix3f &transform, const float lowvalue,
                              const float highvalue);
  /**
   * Calculate the normalization among intersections on a single detector
   * in 1 specific SpectrumInfo/ExperimentInfo
   * @param intersections: intersections
   * @param solid: proton charge
   * @param yValues: diffraction intersection integral and common to sample and background
   * @param signalArray: (output) normalization
   * @param solidBkgd: background proton charge
   * @param bkgdSignalArray: (output) background normalization
   */
  void calcSingleDetectorNorm(const std::vector<std::array<float, 4>> &intersections, const double &solid,
                              std::vector<double> &yValues, histogram_type &h);
private:
  std::vector<float> m_hX, m_kX, m_lX /*, m_eX*/;
};
