#include "mdnorm.h"
#include "LoadEventWorkspace.h"
#include "LoadEventWorkspace2.h"
#include "LoadExtrasWorkspace.h"
#include "LoadFluxWorkspace.h"
#include "LoadSolidAngleWorkspace.h"
#include "calcDiffractionIntersectionIntegral.h"
#include "calculateIntersections.h"
#include "histogram.h"

#include <Eigen/Core>
#include <highfive/highfive.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

class BinMD {
public:
  static constexpr int simd_size = 8;
  BinMD(Eigen::Index cols) {
    std::cout << cols << "\n";
#pragma omp parallel
    {
      vf.resize(simd_size, cols);
      vf2.resize(1, cols);
    }
  }
  void operator()(const Eigen::Matrix<float, 3, Eigen::Dynamic> &transforms,
                  const Eigen::Matrix<float, Eigen::Dynamic, 3> &events, histogram_type &h) {
    using boost::histogram::weight;
#pragma omp parallel for
    for (Eigen::Index i = 0; i < events.rows() - simd_size; i += simd_size) {
      vf = events.block<simd_size, 3>(i, 0) * transforms;
      for (Eigen::Index k = 0; k < vf.rows(); k += 3) {
        for (int j = 0; j < simd_size; ++j) {
          h(vf(j, k), vf(j, k + 1), vf(j, k + 2));//, weight(events(i + j, 0)));
        }
      }
    }
#pragma omp parallel for
    for (Eigen::Index i = events.rows() - events.rows() % simd_size; i < events.rows(); ++i) {
      vf2 =  events.block<1, 3>(i, 0) * transforms;
      for (Eigen::Index j = 0; j < cf2.cols(); j += 3) {
        h(vf2[j], vf2[j + 1], vf2[j + 2]);//, weight(events(i, 0)));
      }
    }
  }

private:
  static Eigen::Matrix<float, simd_size, Eigen::Dynamic> vf;
#pragma omp threadprivate(vf)
  static Eigen::Matrix<float, 1, Eigen::Dynamic> vf2;
#pragma omp threadprivate(vf2)
};

Eigen::Matrix<float, BinMD::simd_size, Eigen::Dynamic> BinMD::vf;
Eigen::Matrix<float, 1, Eigen::Dynamic> BinMD::vf2;

void mdnorm(parameters &params, histogram_type &signal, histogram_type& h) {
  using namespace boost::histogram;
  using reg = axis::regular<float>;

  std::vector<float> hX, kX, lX;
  {
    auto &axis = std::get<0>(params.axes);
    for (auto &&x : axis) {
      hX.push_back(x.lower());
    }
    hX.push_back(axis.bin(axis.size() - 1).upper());
  }
  {
    auto &axis = std::get<1>(params.axes);
    for (auto &&x : axis) {
      kX.push_back(x.lower());
    }
    kX.push_back(axis.bin(axis.size() - 1).upper());
  }
  {
    auto &axis = std::get<2>(params.axes);
    for (auto &&x : axis) {
      lX.push_back(x.lower());
    }
    lX.push_back(axis.bin(axis.size() - 1).upper());
  }

  MDNorm doctest(hX, kX, lX);

  signal = make_histogram_with(dense_storage<accumulator_type>(), std::get<0>(params.axes), std::get<1>(params.axes),
                               std::get<2>(params.axes));

  h = make_histogram_with(dense_storage<accumulator_type>(), std::get<0>(params.axes), std::get<1>(params.axes),
                          std::get<2>(params.axes));

  LoadSolidAngleWorkspace solidAngle(params.solidAngleFilename);
  const std::unordered_map<int32_t, size_t> solidAngDetToIdx = solidAngle.getSolidAngDetToIdx();
  const std::vector<std::vector<double>> solidAngleWS = solidAngle.getSolidAngleValues();

  LoadFluxWorkspace flux(params.fluxFilename);
  const reg integrFlux_x = flux.getFluxAxis();
  const std::vector<std::vector<double>> integrFlux_y = flux.getFluxValues();
  const std::unordered_map<int32_t, size_t> fluxDetToIdx = flux.getFluxDetToIdx();
  const size_t ndets = flux.getNDets();

  auto rot_filename = std::string(params.eventPrefix).append(std::to_string(params.eventMin)).append("_extra_params.hdf5");
  LoadExtrasWorkspace extras(rot_filename);
  // const std::vector<Eigen::Matrix3f> symm = {Eigen::Matrix3f::Identity()};
  const std::vector<Eigen::Matrix3f> symm = extras.getSymmMatrices();
  const Eigen::Matrix3f m_UB = extras.getUBMatrix();
  const std::vector<bool> skip_dets = extras.getSkipDets();

  auto event_filename = std::string(params.eventPrefix).append(std::to_string(params.eventMin)).append("_BEFORE_MDNorm.nxs");

  LoadEventWorkspace eventWS(event_filename);
  const std::vector<float> lowValues = eventWS.getLowValues();
  const std::vector<float> highValues = eventWS.getHighValues();
  const std::vector<float> thetaValues = eventWS.getThetaValues();
  const std::vector<float> phiValues = eventWS.getPhiValues();
  const std::vector<int> detIDs = eventWS.getDetIDs();

  std::vector<Eigen::Matrix3f> transforms2;
  for (const Eigen::Matrix3f &op : symm) {
    Eigen::Matrix3f transform = m_UB * op * params.W;
    transforms2.push_back(transform.inverse());
  }

  Eigen::Matrix<float, 3, Eigen::Dynamic> transforms3(3, symm.size() * 3);
  for (size_t i = 0; i < transforms2.size(); ++i) {
    transforms3.block<3, 3>(0, i * 3) = transforms2[i].transpose();
  }
  BinMD binMD(transforms3.cols());

  std::vector<int> idx;
  std::vector<float> momentum;
  std::vector<Eigen::Vector3f> intersections;
  std::vector<float> xValues;
  std::vector<double> yValues;
  std::vector<Eigen::Matrix3f> transforms;
  Eigen::Matrix<float, Eigen::Dynamic, 3> events;
  std::vector<int> boxType;
  Eigen::Matrix<double, Eigen::Dynamic, 6> boxExtents;
  Eigen::Matrix<double, Eigen::Dynamic, 2> boxSignal;
  Eigen::Matrix<uint64_t, Eigen::Dynamic, 2> boxEventIndex;


  std::cout << params.eventStart << " " <<params.eventStop << std::endl;
  for (int file_num = params.eventMin + params.eventStart; file_num <= params.eventMin + params.eventStop; ++file_num) {
    auto rot_filename_changes =
        std::string(params.eventPrefix).append(std::to_string(file_num)).append("_extra_params.hdf5");
    LoadExtrasWorkspace extras_changes(rot_filename_changes);
    Eigen::Matrix3f rotMatrix = extras_changes.getRotMatrix();

    transforms.clear();
    for (const Eigen::Matrix3f &op : symm) {
      Eigen::Matrix3f transform = rotMatrix * m_UB * op * params.W;
      transforms.push_back(transform.inverse());
    }

    auto event_filename_changes =
        std::string(params.eventPrefix).append(std::to_string(file_num)).append("_events.nxs");
    LoadEventWorkspace2 eventWS_changes(event_filename_changes);
    const double protonCharge = eventWS_changes.getProtonCharge();

    auto startt = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2) private(idx, momentum, intersections, xValues, yValues)
    for (const Eigen::Matrix3f &op : transforms) {
      for (size_t i = 0; i < ndets; ++i) {
        if (skip_dets[i])
          continue;

        int32_t detID = detIDs[i];
        // get the flux spectrum number: this is for diffraction only!
        size_t wsIdx = 0;
        if (auto index = fluxDetToIdx.find(detID); index != fluxDetToIdx.end())
          wsIdx = index->second;
        else // masked detector in flux, but not in input workspace
          continue;

        doctest.calculateIntersections(signal, idx, momentum, intersections, thetaValues[i], phiValues[i], op,
                                       lowValues[i], highValues[i]);

        if (intersections.empty())
          continue;

        // Get solid angle for this contribution
        const double solid_angle_factor = solidAngleWS[solidAngDetToIdx.find(detID)->second][0];
        double solid = protonCharge * solid_angle_factor;

        calcDiffractionIntersectionIntegral(idx, momentum, intersections, xValues, yValues, integrFlux_x, integrFlux_y,
                                            wsIdx);

        doctest.calcSingleDetectorNorm(idx, xValues, intersections, solid, yValues, signal);
      }
    }
    auto stopt = std::chrono::high_resolution_clock::now();
    double duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
    std::cout << " MDNorm time: " << duration_total << "s\n";

    startt = std::chrono::high_resolution_clock::now();
    eventWS_changes.updateEvents(events);
    eventWS_changes.updateBoxType(boxType);
    eventWS_changes.updateExtents(boxExtents);
    eventWS_changes.updateSignal(boxSignal);
    eventWS_changes.updateEventIndex(boxEventIndex);
    stopt = std::chrono::high_resolution_clock::now();
    duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
    std::cout << " updateEvents time: " << duration_total << "s\n";

    startt = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < boxType.size(); ++i) {
      if (boxType[i] == 1 && boxEventIndex(i, 1) != 0) {
        Eigen::Vector3f vi(boxExtents(i, 0), boxExtents(i, 2), boxExtents(i, 4));
        Eigen::Vector3f vi2(boxExtents(i, 1), boxExtents(i, 3), boxExtents(i, 5));
        for (const Eigen::Matrix3f &op : transforms2) {
          const Eigen::Vector3f vf = op * vi;
          const Eigen::Vector3f vf2 = op * vi2;

          Eigen::Vector3i startIdx;
          bool singleBox = true;
          for (int j = 0; j < 3; ++j) {
            startIdx[j] = h.axis(j).index(vf[j]);
            const auto endIdx = h.axis(j).index(vf2[j]);
            if (startIdx[j] != endIdx) {
              singleBox = false;
              continue;
            }
          }
          if (singleBox) {
            // h(vf[0], vf[1], vf[2], weight(boxSignal(i, 0)));
            h.at(startIdx[0], startIdx[1], startIdx[2]) += boxSignal(i, 0);
          } else {
            const size_t end = boxEventIndex(i, 0) + boxEventIndex(i, 1);
            for (size_t j = boxEventIndex(i, 0); j < end; ++j) {
              Eigen::Vector3f vff = op * events.block<1, 3>(j, 5).transpose();
              h(vff[0], vff[1], vff[2], weight(events(j, 0)));
            }
          }
        }
      }
    }

    /*eventWS_changes.updateEvents(events);
    stopt = std::chrono::high_resolution_clock::now();
    duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
    std::cout << " updateEvents time: " << duration_total << "s\n";

    startt = std::chrono::high_resolution_clock::now();
    binMD(transforms3, events, h);
    stopt = std::chrono::high_resolution_clock::now();
    duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
    std::cout << " BinMD time: " << duration_total << "s\n";
  }
}
