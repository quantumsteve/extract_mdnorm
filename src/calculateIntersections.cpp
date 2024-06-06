#include "calculateIntersections.h"

#include <algorithm>
#include <cassert>
#include <numeric>

// function to  compare two intersections (h,k,l,Momentum) by Momentum
static bool compareMomentum(const std::array<float, 4> &v1, const std::array<float, 4> &v2) {
  return (v1[3] < v2[3]);
}

MDNorm::MDNorm(const std::vector<float> &hX, const std::vector<float> &kX, const std::vector<float> &lX)
    : m_hX(hX), m_kX(kX), m_lX(lX), m_transformation(Eigen::Matrix3f::Identity()) {}

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
void MDNorm::calculateIntersections(std::vector<std::array<float, 4>> &intersections, const float theta,
                                    const float phi, const Eigen::Matrix3f &transform, float lowvalue,
                                    float highvalue) {
  Eigen::Vector3f qout(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
  Eigen::Vector3f qin(0., 0., 1.);

  qout = transform * qout;
  qin = transform * qin;
  // if (convention == "Crystallography") {
  //  qout *= -1;
  //  qin *= -1;
  //}
  float kfmin, kfmax, kimin, kimax;
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

  float hStart = qin[0] * kimin - qout[0] * kfmin, hEnd = qin[0] * kimax - qout[0] * kfmax;
  float kStart = qin[1] * kimin - qout[1] * kfmin, kEnd = qin[1] * kimax - qout[1] * kfmax;
  float lStart = qin[2] * kimin - qout[2] * kfmin, lEnd = qin[2] * kimax - qout[2] * kfmax;

  float eps = 1e-10;
  auto hNBins = m_hX.size();
  auto kNBins = m_kX.size();
  auto lNBins = m_lX.size();
  intersections.clear();
  intersections.reserve(hNBins + kNBins + lNBins + 2);

  // calculate intersections with planes perpendicular to h
  if (fabs(hStart - hEnd) > eps) {
    float fmom = (kfmax - kfmin) / (hEnd - hStart);
    float fk = (kEnd - kStart) / (hEnd - hStart);
    float fl = (lEnd - lStart) / (hEnd - hStart);
    for (size_t i = 0; i < hNBins; i++) {
      float hi = m_hX[i];
      if (((hStart - hi) * (hEnd - hi) < 0)) {
        // if hi is between hStart and hEnd, then ki and li will be between
        // kStart, kEnd and lStart, lEnd and momi will be between kfmin and
        // kfmax
        float ki = fk * (hi - hStart) + kStart;
        float li = fl * (hi - hStart) + lStart;
        if ((ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1]) && (li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          float momi = fmom * (hi - hStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
      }
    }
  }
  // calculate intersections with planes perpendicular to k
  if (fabs(kStart - kEnd) > eps) {
    float fmom = (kfmax - kfmin) / (kEnd - kStart);
    float fh = (hEnd - hStart) / (kEnd - kStart);
    float fl = (lEnd - lStart) / (kEnd - kStart);
    for (size_t i = 0; i < kNBins; i++) {
      float ki = m_kX[i];
      if (((kStart - ki) * (kEnd - ki) < 0)) {
        // if ki is between kStart and kEnd, then hi and li will be between
        // hStart, hEnd and lStart, lEnd and momi will be between kfmin and
        // kfmax
        float hi = fh * (ki - kStart) + hStart;
        float li = fl * (ki - kStart) + lStart;
        if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1]) && (li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          float momi = fmom * (ki - kStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
      }
    }
  }

  // calculate intersections with planes perpendicular to l
  if (fabs(lStart - lEnd) > eps) {
    float fmom = (kfmax - kfmin) / (lEnd - lStart);
    float fh = (hEnd - hStart) / (lEnd - lStart);
    float fk = (kEnd - kStart) / (lEnd - lStart);

    for (size_t i = 0; i < lNBins; i++) {
      float li = m_lX[i];
      if (((lStart - li) * (lEnd - li) < 0)) {
        float hi = fh * (li - lStart) + hStart;
        float ki = fk * (li - lStart) + kStart;
        if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1]) && (ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1])) {
          float momi = fmom * (li - lStart) + kfmin;
          intersections.push_back({{hi, ki, li, momi}});
        }
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
 * Calculate the points of intersection for the given detector with cuboid
 * surrounding the detector position in HKL
 * @param intersections A list of intersections in HKL space
 * @param theta Polar angle withd detector
 * @param phi Azimuthal angle with detector
 * @param transform Matrix to convert frm Q_lab to HKL (2Pi*R *UB*W*SO)^{-1}
 * @param lowvalue The lowest momentum or energy transfer for the trajectory
 * @param highvalue The highest momentum or energy transfer for the trajectory
 */
void MDNorm::calculateIntersections(histogram_type &h, std::vector<int> &idx, std::vector<float> &momentum,
                                    std::vector<std::array<float, 3>> &intersections, const float theta,
                                    const float phi, const Eigen::Matrix3f &transform, const float lowvalue,
                                    const float highvalue) {
  Eigen::Vector3f qout(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
  Eigen::Vector3f qin(0.f, 0.f, 1.f);

  qout = transform * qout;
  qin = transform * qin;

  float kimin = lowvalue;
  float kimax = highvalue;
  float kfmin = kimin;
  float kfmax = kimax;

  auto hNBins = static_cast<int64_t>(m_hX.size());
  auto kNBins = static_cast<int64_t>(m_kX.size());
  auto lNBins = static_cast<int64_t>(m_lX.size());

  float hStart = qin[0] * kimin - qout[0] * kfmin;
  float hEnd = qin[0] * kimax - qout[0] * kfmax;

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

  float kStart = qin[1] * kimin - qout[1] * kfmin;
  float kEnd = qin[1] * kimax - qout[1] * kfmax;

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

  float lStart = qin[2] * kimin - qout[2] * kfmin;
  float lEnd = qin[2] * kimax - qout[2] * kfmax;
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
  momentum.clear();
  momentum.reserve(hNBins + kNBins + lNBins + 2);

  // calculate intersections with planes perpendicular to h
  {
    float fmom = (kfmax - kfmin) / (hEnd - hStart);
    float fk = (kEnd - kStart) / (hEnd - hStart);
    float fl = (lEnd - lStart) / (hEnd - hStart);
    for (int i = hStartIdx; i < hEndIdx; ++i) {
      float hi = m_hX[i];
      // if hi is between hStart and hEnd, then ki and li will be between
      // kStart, kEnd and lStart, lEnd and momi will be between kfmin and
      // kfmax
      float ki = fk * (hi - hStart) + kStart;
      if ((ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1])) {
        float li = fl * (hi - hStart) + lStart;
        if ((li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          float momi = fmom * (hi - hStart) + kfmin;
          momentum.push_back(momi);
          intersections.push_back({{hi, ki, li}});
        }
      }
    }
  }

  // calculate intersections with planes perpendicular to k
  {
    float fmom = (kfmax - kfmin) / (kEnd - kStart);
    float fh = (hEnd - hStart) / (kEnd - kStart);
    float fl = (lEnd - lStart) / (kEnd - kStart);
    for (auto i = kStartIdx; i < kEndIdx; ++i) {
      float ki = m_kX[i];
      // if ki is between kStart and kEnd, then hi and li will be between
      // hStart, hEnd and lStart, lEnd and momi will be between kfmin and
      // kfmax
      float hi = fh * (ki - kStart) + hStart;
      if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1])) {
        float li = fl * (ki - kStart) + lStart;
        if ((li >= m_lX[0]) && (li <= m_lX[lNBins - 1])) {
          float momi = fmom * (ki - kStart) + kfmin;
          momentum.push_back(momi);
          intersections.push_back({{hi, ki, li}});
        }
      }
    }
  }

  // calculate intersections with planes perpendicular to l
  {
    float fmom = (kfmax - kfmin) / (lEnd - lStart);
    float fh = (hEnd - hStart) / (lEnd - lStart);
    float fk = (kEnd - kStart) / (lEnd - lStart);

    for (auto i = lStartIdx; i < lEndIdx; ++i) {
      float li = m_lX[i];
      float hi = fh * (li - lStart) + hStart;
      if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1])) {
        float ki = fk * (li - lStart) + kStart;
        if ((ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1])) {
          float momi = fmom * (li - lStart) + kfmin;
          momentum.push_back(momi);
          intersections.push_back({{hi, ki, li}});
        }
      }
    }
  }

  // endpoints
  if ((hStart >= m_hX[0]) && (hStart <= m_hX[hNBins - 1]) && (kStart >= m_kX[0]) && (kStart <= m_kX[kNBins - 1]) &&
      (lStart >= m_lX[0]) && (lStart <= m_lX[lNBins - 1])) {
    momentum.push_back(kfmin);
    intersections.push_back({{hStart, kStart, lStart}});
  }
  if ((hEnd >= m_hX[0]) && (hEnd <= m_hX[hNBins - 1]) && (kEnd >= m_kX[0]) && (kEnd <= m_kX[kNBins - 1]) &&
      (lEnd >= m_lX[0]) && (lEnd <= m_lX[lNBins - 1])) {
    momentum.push_back(kfmax);
    intersections.push_back({{hEnd, kEnd, lEnd}});
  }

  // sort intersections by final momentum
  idx.resize(intersections.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&momentum](int i1, int i2) { return momentum[i1] < momentum[i2]; });
}

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
void MDNorm::calcSingleDetectorNorm(const std::vector<int> &idx, const std::vector<float> &momentum,
                                    const std::vector<std::array<float, 3>> &intersections, const double solid,
                                    std::vector<double> &yValues, histogram_type &h) {

  auto idxBegin = idx.begin();
  for (auto it = idxBegin + 1; it != idx.end(); ++it) {
    // The full vector isn't used so compute only what is necessary
    // If the difference between 2 adjacent intersection is trivial, no
    // intersection normalization is to be calculated
    // diffraction
    double delta = momentum[*it] - momentum[*(it - 1)];
    double eps = 1e-7;
    if (delta < eps)
      continue; // Assume zero contribution if difference is small

    const auto &curIntSec = intersections[*it];
    const auto &prevIntSec = intersections[*(it - 1)];
    // Average between two intersections for final position
    // Find the coordiate of the new position after transformation
    Eigen::Map<const Eigen::Vector3f> pos1(curIntSec.data());
    Eigen::Map<const Eigen::Vector3f> pos2(prevIntSec.data());
    Eigen::Vector3f posNew = m_transformation * 0.5 * (pos1 + pos2);

    // Diffraction
    // index of the current intersection
    auto k = static_cast<size_t>(std::distance(idxBegin, it));
    // signal = integral between two consecutive intersections
    double signal = (yValues[k] - yValues[k - 1]) * solid;

    using boost::histogram::weight;
    // Set to output
    h(posNew[0], posNew[1], posNew[2], weight(signal));
  }
  return;
}
