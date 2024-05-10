#pragma once

#include <Eigen/Core>
#include <boost/histogram.hpp>

#include <array>
#include <atomic>
#include <iostream>
#include <vector>

// function to  compare two intersections (h,k,l,Momentum) by Momentum
inline bool compareMomentum(const std::array<double, 4> &v1, const std::array<double, 4> &v2) {
  return (v1[3] < v2[3]);
}

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
  void calculateIntersections(std::vector<std::array<double, 4>> &intersections, double theta, double phi,
                              const Eigen::Matrix3d &transform, double lowvalue, double highvalue);
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
  template <typename histogram>
  void calculateIntersections(histogram &h, std::vector<std::array<double, 4>> &intersections, const double theta,
                              const double phi, const Eigen::Matrix3d &transform, double lowvalue, double highvalue) {
    Eigen::Vector3d qout(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
    Eigen::Vector3d qin(0., 0., 1.);

    qout = transform * qout;
    qin = transform * qin;
    // if (convention == "Crystallography") {
    //  qout *= -1;
    //  qin *= -1;
    //}
    double kfmin, kfmax, kimin, kimax;
    // if (m_diffraction) {
    kimin = lowvalue;
    kimax = highvalue;
    kfmin = kimin;
    kfmax = kimax;
    //} else {
    //  kimin = std::sqrt(energyToK * m_Ei);
    //  kimax = kimin;
    //  kfmin = std::sqrt(energyToK * (m_Ei - highvalue));
    //  kfmax = std::sqrt(energyToK * (m_Ei - lowvalue));
    //}

    auto hNBins = static_cast<int64_t>(m_hX.size());
    auto kNBins = static_cast<int64_t>(m_kX.size());
    auto lNBins = static_cast<int64_t>(m_lX.size());

    double hStart = qin[0] * kimin - qout[0] * kfmin;
    double hEnd = qin[0] * kimax - qout[0] * kfmax;

    auto hStartIdx = h.axis(0).index(hStart);
    auto hEndIdx = h.axis(0).index(hEnd);
    if (hStartIdx > hEndIdx)
      std::swap(hStartIdx, hEndIdx);
    ++hStartIdx;
    hEndIdx = std::max(hStartIdx, hEndIdx);

    if (hStartIdx > 0) {
      if (auto hi = m_hX[hStartIdx - 1]; (hStart - hi) * (hEnd - hi) < 0) {
        --hStartIdx;
      }
    }
    if (hEndIdx < hNBins) {
      if (auto hi = m_hX[hEndIdx]; (hStart - hi) * (hEnd - hi) < 0) {
        ++hEndIdx;
      }
    }

    double kStart = qin[1] * kimin - qout[1] * kfmin;
    double kEnd = qin[1] * kimax - qout[1] * kfmax;

    auto kStartIdx = h.axis(1).index(kStart);
    auto kEndIdx = h.axis(1).index(kEnd);
    if (kStartIdx > kEndIdx)
      std::swap(kStartIdx, kEndIdx);
    ++kStartIdx;
    kEndIdx = std::max(kStartIdx, kEndIdx);

    if (kStartIdx > 0) {
      if (auto ki = m_kX[kStartIdx - 1]; (kStart - ki) * (kEnd - ki) < 0) {
        --kStartIdx;
      }
    }
    if (kEndIdx < kNBins) {
      if (auto ki = m_kX[kEndIdx]; (kStart - ki) * (kEnd - ki) < 0) {
        ++kEndIdx;
      }
    }

    double lStart = qin[2] * kimin - qout[2] * kfmin;
    double lEnd = qin[2] * kimax - qout[2] * kfmax;
    auto lStartIdx = h.axis(2).index(lStart);
    auto lEndIdx = h.axis(2).index(lEnd);
    if (lStartIdx > lEndIdx)
      std::swap(lStartIdx, lEndIdx);
    ++lStartIdx;
    lEndIdx = std::max(lStartIdx, lEndIdx);

    if (lStartIdx > 0) {
      if (auto li = m_lX[lStartIdx - 1]; (lStart - li) * (lEnd - li) < 0) {
        --lStartIdx;
      }
    }
    if (lEndIdx < lNBins) {
      if (auto li = m_lX[lEndIdx]; (lStart - li) * (lEnd - li) < 0) {
        ++lEndIdx;
      }
    }

    intersections.clear();
    intersections.reserve(hNBins + kNBins + lNBins + 2);

    // calculate intersections with planes perpendicular to h
    {
      double fmom = (kfmax - kfmin) / (hEnd - hStart);
      double fk = (kEnd - kStart) / (hEnd - hStart);
      double fl = (lEnd - lStart) / (hEnd - hStart);
      for (int i = hStartIdx; i < hEndIdx; ++i) {
        double hi = m_hX[i];
        // if hi is between hStart and hEnd, then ki and li will be between
        // kStart, kEnd and lStart, lEnd and momi will be between kfmin and
        // kfmax
        double ki = fk * (hi - hStart) + kStart;
        double li = fl * (hi - hStart) + lStart;
        if ((ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1]) && (li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          double momi = fmom * (hi - hStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
      }
    }

    // calculate intersections with planes perpendicular to k
    {
      double fmom = (kfmax - kfmin) / (kEnd - kStart);
      double fh = (hEnd - hStart) / (kEnd - kStart);
      double fl = (lEnd - lStart) / (kEnd - kStart);
      for (auto i = kStartIdx; i < kEndIdx; ++i) {
        double ki = m_kX[i];
        // if ki is between kStart and kEnd, then hi and li will be between
        // hStart, hEnd and lStart, lEnd and momi will be between kfmin and
        // kfmax
        double hi = fh * (ki - kStart) + hStart;
        double li = fl * (ki - kStart) + lStart;
        if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1]) && (li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          double momi = fmom * (ki - kStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
      }
    }

    // calculate intersections with planes perpendicular to l
    {
      double fmom = (kfmax - kfmin) / (lEnd - lStart);
      double fh = (hEnd - hStart) / (lEnd - lStart);
      double fk = (kEnd - kStart) / (lEnd - lStart);

      for (auto i = lStartIdx; i < lEndIdx; ++i) {
        double li = m_lX[i];
        double hi = fh * (li - lStart) + hStart;
        double ki = fk * (li - lStart) + kStart;
        if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1]) && (ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1])) {
          double momi = fmom * (li - lStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
      }
    }

    // endpoints
    if ((hStart >= m_hX[0]) && (hStart <= m_hX[hNBins - 1]) && (kStart >= m_kX[0]) && (kStart <= m_kX[kNBins - 1]) &&
        (lStart >= m_lX[0]) && (lStart <= m_lX[lNBins - 1])) {
      intersections.push_back({{hStart, kStart, lStart, kfmin}});
    }
    if ((hEnd >= m_hX[0]) && (hEnd <= m_hX[hNBins - 1]) && (kEnd >= m_kX[0]) && (kEnd <= m_kX[kNBins - 1]) &&
        (lEnd >= m_lX[0]) && (lEnd <= m_lX[lNBins - 1])) {
      intersections.push_back({{hEnd, kEnd, lEnd, kfmax}});
    }

    // sort intersections by final momentum
    std::sort(intersections.begin(), intersections.end(), compareMomentum);
  }
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
  template <typename histogram>
  void calcSingleDetectorNorm(const std::vector<std::array<double, 4>> &intersections, const double &solid,
                              std::vector<double> &yValues, const size_t &vmdDims, std::vector<float> &pos,
                              std::vector<float> &posNew, histogram &h) {

    auto intersectionsBegin = intersections.begin();
    for (auto it = intersectionsBegin + 1; it != intersections.end(); ++it) {

      const auto &curIntSec = *it;
      const auto &prevIntSec = *(it - 1);

      // The full vector isn't used so compute only what is necessary
      // If the difference between 2 adjacent intersection is trivial, no
      // intersection normalization is to be calculated
      // diffraction
      double delta = curIntSec[3] - prevIntSec[3];
      double eps = 1e-7;
      if (delta < eps)
        continue; // Assume zero contribution if difference is small

      // Average between two intersections for final position
      // [Task 89] Sample and background have same 'pos[]'
      std::transform(curIntSec.data(), curIntSec.data() + vmdDims, prevIntSec.data(), pos.begin(),
                     [](const double rhs, const double lhs) { return static_cast<float>(0.5 * (rhs + lhs)); });

      // Diffraction
      // index of the current intersection
      auto k = static_cast<size_t>(std::distance(intersectionsBegin, it));
      // signal = integral between two consecutive intersections
      double signal = (yValues[k] - yValues[k - 1]) * solid;

      // Find the coordiate of the new position after transformation

      // m_transformation.multiplyPoint(pos, posNew); (identify matrix)
      std::copy(std::begin(pos), std::end(pos), std::begin(posNew));

      // [Task 89] Is linIndex common to both sample and background?
      // size_t linIndex = this->getLinearIndexAtCoord(posNew.data());

      // std::cout << posNew[0] << " " << posNew[1] << " " << posNew[2] << " " << linIndex << std::endl;

      // if (linIndex == size_t(-1))
      //  continue; // not found

      using namespace boost::histogram;
      // Set to output
      // set the calculated signal to
      h(posNew[0], posNew[1], posNew[2], weight(signal));
    }
    return;
  }

private:
  std::vector<double> m_hX, m_kX, m_lX /*, m_eX*/;
};
