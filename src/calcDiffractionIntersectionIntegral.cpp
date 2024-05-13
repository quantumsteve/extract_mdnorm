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
void calcIntegralsForIntersections(const std::vector<float> &xValues,
                                   const std::vector<std::vector<float>> &integrFlux_x,
                                   const std::vector<std::vector<double>> &integrFlux_y, const size_t sp,
                                   std::vector<double> &yValues) {
  yValues.resize(xValues.size());
  assert(xValues.size() == yValues.size());

  // the x-data from the workspace
  const auto &xData = integrFlux_x[0];
  const double xStart = xData.front();
  const double xEnd = xData.back();

  // the values in integrFlux are expected to be integrals of a non-negative
  // function
  // ie they must make a non-decreasing function
  const auto &yData = integrFlux_y[sp];

  const double yMin = 0.0;
  const double yMax = yData.back();

  size_t nData = xValues.size();
  // all integrals below xStart must be 0
  if (xValues[nData - 1] < xStart) {
    std::fill(yValues.begin(), yValues.end(), yMin);
    return;
  }

  // all integrals above xEnd must be equal to yMax
  if (xValues[0] > xEnd) {
    std::fill(yValues.begin(), yValues.end(), yMax);
    return;
  }

  size_t i = 0;
  // integrals below xStart must be 0
  while (i < nData - 1 && xValues[i] <= xStart) {
    yValues[i] = yMin;
    ++i;
  }

  size_t iMax = nData - 1;
  // integrals above xEnd must be yMax
  while (iMax > i && xValues[iMax] >= xEnd) {
    yValues[iMax] = yMax;
    --iMax;
  }

  using namespace boost::histogram;
  using reg = axis::regular<float>;
  reg axes(xData.size() - 1, xStart, xEnd, "x");
  double inv_step = 1. / (xData[1] - xData[0]);

  for (; i <= iMax; ++i) {
    double xi = xValues[i];
    int j = axes.index(xi);
    // interpolate between the consecutive points
    double x0 = axes.bin(j).lower();
    double y0 = yData[j];
    double y1 = yData[j + 1];
    yValues[i] = std::lerp(y0, y1, (xi - x0) * inv_step);
  }
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
void calcDiffractionIntersectionIntegral(std::vector<std::array<float, 4>> &intersections, std::vector<float> &xValues,
                                         std::vector<double> &yValues,
                                         const std::vector<std::vector<float>> &integrFlux_x,
                                         const std::vector<std::vector<double>> &integrFlux_y, const size_t wsIdx) {
  // -- calculate integrals for the intersection --
  // momentum values at intersections
  auto intersectionsBegin = intersections.begin();
  // copy momenta to xValues
  xValues.resize(intersections.size());
  auto x = xValues.begin();
  for (auto it = intersectionsBegin; it != intersections.end(); ++it, ++x) {
    *x = (*it)[3];
  }
  // calculate integrals at momenta from xValues by interpolating between
  // points in spectrum sp
  // of workspace integrFlux. The result is stored in yValues
  calcIntegralsForIntersections(xValues, integrFlux_x, integrFlux_y, wsIdx, yValues);
}
