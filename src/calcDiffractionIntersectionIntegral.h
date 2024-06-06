#pragma once

#include <boost/histogram.hpp>

#include <array>
#include <cstddef>
#include <vector>

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
                                         const std::vector<std::array<float, 3>> &intersections,
                                         std::vector<float> &xValues, std::vector<double> &yValues,
                                         const boost::histogram::axis::regular<float> &integrFlux_x,
                                         const std::vector<std::vector<double>> &integrFlux_y, const size_t wsIndx);
