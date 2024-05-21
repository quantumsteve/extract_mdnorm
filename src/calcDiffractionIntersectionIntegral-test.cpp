#include "LoadEventWorkspace.h"
#include "LoadExtrasWorkspace.h"
#include "LoadFluxWorkspace.h"
#include "LoadSolidAngleWorkspace.h"
#include "calcDiffractionIntersectionIntegral.h"
#include "calculateIntersections.h"
#include "validation_data_filepath.h"

#include "catch2/catch_all.hpp"
#include <boost/histogram.hpp>
#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>

TEST_CASE("calculateIntersections") {
  SECTION("t0") {

    using namespace boost::histogram;
    using reg = axis::regular<float>;
    std::tuple<reg, reg, reg> axes{reg(200, -10., 10., "x"), reg(200, -10., 10., "y"), reg(1, -0.1, 0.1, "z")};

    std::vector<float> hX, kX, lX;
    {
      auto &axis = std::get<0>(axes);
      for (auto &&x : axis) {
        hX.push_back(x.lower());
      }
      hX.push_back(axis.bin(axis.size() - 1).upper());
    }
    {
      auto &axis = std::get<1>(axes);
      for (auto &&x : axis) {
        kX.push_back(x.lower());
      }
      kX.push_back(axis.bin(axis.size() - 1).upper());
    }
    {
      auto &axis = std::get<2>(axes);
      for (auto &&x : axis) {
        lX.push_back(x.lower());
      }
      lX.push_back(axis.bin(axis.size() - 1).upper());
    }

    MDNorm doctest(hX, kX, lX);

    LoadExtrasWorkspace extras(ROT_NXS);
    const Eigen::Matrix3f rotMatrix = extras.getRotMatrix();
    const std::vector<Eigen::Matrix3f> symm = extras.getSymmMatrices();
    const Eigen::Matrix3f m_UB = extras.getUBMatrix();
    const std::vector<bool> skip_dets = extras.getSkipDets();

    Eigen::Matrix3f m_W;
    m_W << 1.f, 1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 0.f, 1.f;

    LoadSolidAngleWorkspace solidAngle(SA_NXS);
    const std::unordered_map<int32_t, size_t> solidAngDetToIdx = solidAngle.getSolidAngDetToIdx();
    const std::vector<std::vector<double>> solidAngleWS = solidAngle.getSolidAngleValues();

    LoadFluxWorkspace flux(FLUX_NXS);
    const reg integrFlux_x = flux.getFluxAxis();
    const std::vector<std::vector<double>> integrFlux_y = flux.getFluxValues();
    const std::unordered_map<int32_t, size_t> fluxDetToIdx = flux.getFluxDetToIdx();
    const size_t ndets = flux.getNDets();

    LoadEventWorkspace eventWS(EVENT_NXS);

    const std::vector<float> lowValues = eventWS.getLowValues();
    const std::vector<float> highValues = eventWS.getHighValues();
    const double protonCharge = eventWS.getProtonCharge();
    const std::vector<float> thetaValues = eventWS.getThetaValues();
    const std::vector<float> phiValues = eventWS.getPhiValues();
    const std::vector<int> detIDs = eventWS.getDetIDs();
    const std::vector<std::array<double, 8>> events = eventWS.getEvents();

    auto signal = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                      std::get<1>(axes), std::get<2>(axes));

    std::vector<Eigen::Matrix3f> transforms;
    for (const Eigen::Matrix3f &op : symm) {
      Eigen::Matrix3f transform = rotMatrix * m_UB * op * m_W;
      transforms.push_back(transform.inverse());
    }

    std::vector<Eigen::Matrix3f> transforms2;
    for (const Eigen::Matrix3f &op : symm) {
      Eigen::Matrix3f transform = m_UB * op * m_W;
      transforms2.push_back(transform.inverse());
    }

    std::vector<std::array<float, 4>> intersections;
    std::vector<float> xValues;
    std::vector<double> yValues;
    auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2) private(intersections, xValues, yValues)
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

        doctest.calculateIntersections(signal, intersections, thetaValues[i], phiValues[i], op, lowValues[i],
                                       highValues[i]);

        if (intersections.empty())
          continue;

        // Get solid angle for this contribution
        const double solid_angle_factor = solidAngleWS[solidAngDetToIdx.find(detID)->second][0];
        double solid = protonCharge * solid_angle_factor;

        calcDiffractionIntersectionIntegral(intersections, xValues, yValues, integrFlux_x, integrFlux_y, wsIdx);

        doctest.calcSingleDetectorNorm(intersections, solid, yValues, signal);
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stop - start).count();
    std::cout << " time: " << duration_total << "s\n";

    HighFive::File norm_file(NORM_NXS, HighFive::File::ReadOnly);
    HighFive::Group norm_group = norm_file.getGroup("MDHistoWorkspace");
    HighFive::Group norm_group2 = norm_group.getGroup("data");
    HighFive::DataSet norm_dataset = norm_group2.getDataSet("signal");
    auto dims = norm_dataset.getDimensions();
    REQUIRE(dims.size() == 3);
    REQUIRE(dims[0] == 1);
    REQUIRE(dims[1] == 200);
    REQUIRE(dims[2] == 200);
    std::vector<std::vector<std::vector<double>>> data;
    norm_dataset.read(data);

    auto &data2d = data[0];
    double max_signal = *std::max_element(signal.begin(), signal.end());

    double ref_max{0.};
    for (size_t i = 0; i < dims[1]; ++i) {
      for (size_t j = 0; j < dims[2]; ++j) {
        REQUIRE_THAT(data2d[i][j], Catch::Matchers::WithinAbs(signal.at(j, i, 0), 5.e+05));
        ref_max = std::max(ref_max, data2d[i][j]);
      }
    }
    REQUIRE_THAT(max_signal, Catch::Matchers::WithinAbs(ref_max, 2.e+04));

    auto h = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                 std::get<1>(axes), std::get<2>(axes));
    start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for collapse(2)
    for (const Eigen::Matrix3f &op : transforms2) {
      for (auto &val : events) {
        Eigen::Vector3f v(val[5], val[6], val[7]);
        v = op * v;
        h(v[0], v[1], v[2], weight(val[0]));
      }
    }
    stop = std::chrono::high_resolution_clock::now();
    duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stop - start).count();
    std::cout << " time: " << duration_total << "s\n";

    std::vector<double> out;
    for (auto &&x : indexed(h))
      out.push_back(*x);

    std::vector<double> meow;
    for (auto &&x : indexed(signal))
      meow.push_back(*x);

    std::ofstream out_strm("meow.txt");
    for (size_t i = 0; i < out.size(); ++i)
      out_strm << out[i] / meow[i] << '\n';
  }
}
