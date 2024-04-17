#pragma once

#include <Eigen/Core>

#include <array>
#include <vector>

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

class MDNorm
{
public:	
  MDNorm(const std::vector<double> &hX, const std::vector<double> &kX, const std::vector<double> &lX);
  void calculateIntersections(std::vector<std::array<double, 4>> &intersections, double theta,
                              double phi, const Eigen::Matrix3d &transform, double lowvalue,
                              double highvalue);
private:
  std::vector<double> m_hX, m_kX, m_lX, m_eX;
};
