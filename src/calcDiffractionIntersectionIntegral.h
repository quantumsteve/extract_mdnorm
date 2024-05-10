#pragma once

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

void calcDiffractionIntersectionIntegral(std::vector<std::array<float, 4>> &intersections, std::vector<float> &xValues,
                                         std::vector<double> &yValues,
                                         const std::vector<std::vector<float>> &integrFlux_x,
                                         const std::vector<std::vector<double>> &integrFlux_y, const size_t wsIndx);
