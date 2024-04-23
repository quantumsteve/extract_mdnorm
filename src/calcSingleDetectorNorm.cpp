#include "calcSingleDetectorNorm.h"

#include<algorithm>
#include <iostream>
#include <cassert>
/** Uses std::compare_exchange_weak to update the atomic value f = op(f, d)
 * Used to improve parallel scaling in algorithms MDNormDirectSC and MDNormSCD
 * @param f atomic variable being updated
 * @param d second element in binary operation
 * @param op binary operation on elements f and d
 */
template <typename T, typename BinaryOp>
void AtomicOp(std::atomic<T> &f, T d, BinaryOp op) {
  T old = f.load();
  T desired;
  do {
    desired = op(old, d);
  } while (!f.compare_exchange_weak(old, desired));
}

void SetUpIndexMaker(const size_t numDims, size_t *out, const size_t *index_max) {
  // Allocate and start at 1
  for (size_t d = 0; d < numDims; d++)
    out[d] = 1;
 
  for (size_t d = 1; d < numDims; d++)
    out[d] = out[d - 1] * index_max[d - 1];  
}

size_t getLinearIndexAtCoord(const float *coords) {
  // Build up the linear index, dimension by dimension
  size_t linearIndex = 0;
  float m_origin[3] = {-10., -10, -0.1};
  float m_boxLength[3] = {0.1, 0.1, 0.2};
  size_t m_indexMax[3] = {200,200,1};
  size_t m_indexMaker[3]{0};

  SetUpIndexMaker(3, m_indexMaker, m_indexMax);

  for (size_t d = 0; d < 3; d++) {
    float x = coords[d] - m_origin[d];
    auto ix = size_t(x / m_boxLength[d]);
    assert(x>=0.f);
    if (ix >= m_indexMax[d] || (x < 0)) {
      return size_t(-1);
    }
    linearIndex += ix * m_indexMaker[d];
  }
  return linearIndex;
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
void calcSingleDetectorNorm(const std::vector<std::array<double, 4>> &intersections, const double &solid,
                                           std::vector<double> &yValues, const size_t &vmdDims,
                                           std::vector<float> &pos, std::vector<float> &posNew,
                                           std::vector<std::atomic<double>> &signalArray, const double &solidBkgd,
                                           std::vector<std::atomic<double>> &bkgdSignalArray) {

  auto intersectionsBegin = intersections.begin();
  for (auto it = intersectionsBegin + 1; it != intersections.end(); ++it) {

    const auto &curIntSec = *it;
    const auto &prevIntSec = *(it - 1);

    // The full vector isn't used so compute only what is necessary
    // If the difference between 2 adjacent intersection is trivial, no
    // intersection normalization is to be calculated
    double delta, eps;
    // diffraction
    delta = curIntSec[3] - prevIntSec[3];
    eps = 1e-7;
    if (delta < eps)
      continue; // Assume zero contribution if difference is small

    // Average between two intersections for final position
    // [Task 89] Sample and background have same 'pos[]'
    std::transform(curIntSec.data(), curIntSec.data() + vmdDims, prevIntSec.data(), pos.begin(),
                   [](const double rhs, const double lhs) { return static_cast<float>(0.5 * (rhs + lhs)); });
    double signal;
    double bkgdSignal(0.);
    // Diffraction
    // index of the current intersection
    auto k = static_cast<size_t>(std::distance(intersectionsBegin, it));
    // signal = integral between two consecutive intersections
    signal = (yValues[k] - yValues[k - 1]) * solid;
    bkgdSignal = (yValues[k] - yValues[k - 1]) * solidBkgd;

    // Find the coordiate of the new position after transformation
    
    //m_transformation.multiplyPoint(pos, posNew); (identify matrix)
    std::copy(std::begin(pos),std::end(pos),std::begin(posNew));
    

    // [Task 89] Is linIndex common to both sample and background?
    size_t linIndex = getLinearIndexAtCoord(posNew.data());

    //std::cout << posNew[0] << " " << posNew[1] << " " << posNew[2] << " " << linIndex << std::endl;

    if (linIndex == size_t(-1))
      continue; // not found



    // Set to output
    // set the calculated signal to
    AtomicOp(signalArray[linIndex], signal, std::plus<double>());
    // [Task 89]
    AtomicOp(bkgdSignalArray[linIndex], bkgdSignal, std::plus<double>());
  }
  return;
}