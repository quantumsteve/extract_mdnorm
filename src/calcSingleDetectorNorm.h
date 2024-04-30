#pragma once

#include <array>
#include <atomic>
#include <vector>

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
                            std::vector<double> &yValues, const size_t &vmdDims,
                            std::vector<float> &pos, std::vector<float> &posNew,
                            std::vector<std::atomic<double>> &signalArray);