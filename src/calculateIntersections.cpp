#include "calculateIntersections.h"

#include <iomanip>
#include <iostream>

// function to  compare two intersections (h,k,l,Momentum) by Momentum
bool compareMomentum(const std::array<double, 4> &v1, const std::array<double, 4> &v2) { return (v1[3] < v2[3]); }

MDNorm::MDNorm(const std::vector<double> &hX, const std::vector<double> &kX, const std::vector<double> &lX) : m_hX(hX), m_kX(kX), m_lX(lX) {}

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
void MDNorm::calculateIntersections(std::vector<std::array<double, 4>> &intersections, const double theta,
                                    const double phi, const Eigen::Matrix3d &transform, double lowvalue,
                                    double highvalue) {
  Eigen::Vector3d qout(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
  Eigen::Vector3d qin(0., 0., 1.);

  qout = transform * qout;
  qin = transform * qin;
  //if (convention == "Crystallography") {
  //  qout *= -1;
  //  qin *= -1;
  //}
  double kfmin, kfmax, kimin, kimax;
  //if (m_diffraction) {
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

  double hStart = qin[0] * kimin - qout[0] * kfmin, hEnd = qin[0] * kimax - qout[0] * kfmax;
  double kStart = qin[1] * kimin - qout[1] * kfmin, kEnd = qin[1] * kimax - qout[1] * kfmax;
  double lStart = qin[2] * kimin - qout[2] * kfmin, lEnd = qin[2] * kimax - qout[2] * kfmax;

  double eps = 1e-10;
  auto hNBins = m_hX.size();
  auto kNBins = m_kX.size();
  auto lNBins = m_lX.size();
  intersections.clear();
  intersections.reserve(hNBins + kNBins + lNBins + 2);

  // calculate intersections with planes perpendicular to h
  if (fabs(hStart - hEnd) > eps) {
    double fmom = (kfmax - kfmin) / (hEnd - hStart);
    double fk = (kEnd - kStart) / (hEnd - hStart);
    double fl = (lEnd - lStart) / (hEnd - hStart);
    for (size_t i = 0; i < hNBins; i++) {
      double hi = m_hX[i];
      if (((hStart - hi) * (hEnd - hi) < 0)) {
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
  }
  // calculate intersections with planes perpendicular to k
  if (fabs(kStart - kEnd) > eps) {
    double fmom = (kfmax - kfmin) / (kEnd - kStart);
    double fh = (hEnd - hStart) / (kEnd - kStart);
    double fl = (lEnd - lStart) / (kEnd - kStart);
    for (size_t i = 0; i < kNBins; i++) {
      double ki = m_kX[i];
      if (((kStart - ki) * (kEnd - ki) < 0)) {
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
  }

  // calculate intersections with planes perpendicular to l
  if (fabs(lStart - lEnd) > eps) {
    double fmom = (kfmax - kfmin) / (lEnd - lStart);
    double fh = (hEnd - hStart) / (lEnd - lStart);
    double fk = (kEnd - kStart) / (lEnd - lStart);

    for (size_t i = 0; i < lNBins; i++) {
      double li = m_lX[i];
      if (((lStart - li) * (lEnd - li) < 0)) {
        double hi = fh * (li - lStart) + hStart;
        double ki = fk * (li - lStart) + kStart;
        if ((hi >= m_hX[0]) && (hi <= m_hX[hNBins - 1]) && (ki >= m_kX[0]) && (ki <= m_kX[kNBins - 1])) {
          double momi = fmom * (li - lStart) + kfmin;
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
