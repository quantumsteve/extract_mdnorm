#include "calcDiffractionIntersectionIntegral.h"

#include <boost/histogram.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

/**
 * Linearly interpolate between the points in integrFlux at xValues and save the
 * results in yValues.
 * @param xValues :: X-values at which to interpolate
 * @param integrFlux :: A workspace with the spectra to interpolate
 * @param sp :: A workspace index for a spectrum in integrFlux to interpolate.
 * @param yValues :: A vector to save the results.
 */
static void calcIntegralsForIntersections(const std::vector<float> &xValues,
                                   const boost::histogram::axis::regular<float> &integrFlux_x,
                                   const std::vector<std::vector<double>> &integrFlux_y, const size_t sp,
                                   std::vector<double> &yValues) {

  assert(xValues.size() == yValues.size());

  // the x-data from the workspace
  const float xStart = integrFlux_x.bin(0).lower();

  // the values in integrFlux are expected to be integrals of a non-negative
  // function
  // ie they must make a non-decreasing function
  const auto &yData = integrFlux_y[sp];

  const double yMin = 0.0;
  const double yMax = yData.back();

  // all integrals below xStart must be 0
  if (xValues.back() < xStart) {
    std::fill(yValues.begin(), yValues.end(), yMin);
    return;
  }

  const float xEnd = integrFlux_x.bin(integrFlux_x.size()).upper();
  // all integrals above xEnd must be equal to yMax
  if (xValues[0] > xEnd) {
    std::fill(yValues.begin(), yValues.end(), yMax);
    return;
  }

  // integrals below xStart must be 0
  auto it = std::upper_bound(xValues.begin(), xValues.end(), xStart);
  auto i = std::distance(xValues.begin(), it);
  std::fill_n(yValues.begin(), i, yMin);

  it = std::upper_bound(it, xValues.end(), xEnd);
  const auto iMax = std::distance(xValues.begin(), it);
  const float inv_step = 1.f / integrFlux_x.bin(0).width();
  for (; i < iMax; ++i) {
    float xi = xValues[i];
    auto j = integrFlux_x.index(xi);
    // interpolate between the consecutive points
    float x0 = integrFlux_x.bin(j).lower();
    double y0 = yData[j];
    double y1 = yData[j + 1];
    yValues[i] = std::lerp(y0, y1, static_cast<double>((xi - x0) * inv_step));
  }

  std::fill(yValues.begin() + iMax, yValues.end(), yMax);
}

/**
 * Calculate the diffraction MDE's intersection integral of a certain
 * detector/spectru
 * @param intersections: vector of intersections
 * @param xValues: empty vector for X values
 * @param yValues: empty vector of Y values (output)
 * @param integrFlux: integral flux workspace
 * @param wsIdx: workspace index
 */
void calcDiffractionIntersectionIntegral(const std::vector<int> &idx, const std::vector<float> &momentum,
                                         const std::vector<Eigen::Vector3f> &intersections, std::vector<float> &xValues,
                                         std::vector<double> &yValues,
                                         const boost::histogram::axis::regular<float> &integrFlux_x,
                                         const std::vector<std::vector<double>> &integrFlux_y, const size_t wsIdx) {
  // -- calculate integrals for the intersection --
  // copy momenta to xValues
  // xValues.resize(intersections.size());
  yValues.resize(intersections.size());
  // auto x = xValues.begin();
  // for (auto it = idx.begin(); it != idx.end(); ++it, ++x) {
  //  *x = momentum[*it];
  //}
  // calculate integrals at momenta from xValues by interpolating between
  // points in spectrum sp
  // of workspace integrFlux. The result is stored in yValues
  calcIntegralsForIntersections(momentum, integrFlux_x, integrFlux_y, wsIdx, yValues);
}
