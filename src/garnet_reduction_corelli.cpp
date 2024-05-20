#include "calcDiffractionIntersectionIntegral.h"
#include "calculateIntersections.h"

#include "catch2/catch_all.hpp"
#include "validation_data_filepath.h"

#include <boost/histogram.hpp>
#include <boost/math/constants/constants.hpp>
#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

TEST_CASE("calculateIntersections") {
  SECTION("t0") {

    using namespace boost::histogram;
    using reg = axis::regular<float>;
    std::tuple<reg, reg, reg> axes{reg(201, -10.05, 10.05, "x"), reg(201, -10.05, 10.05, "y"),
                                   reg(201, -10.05, 10.05, "z")};

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

    HighFive::File rot_file(GARNET_ROT_NXS, HighFive::File::ReadOnly);
    HighFive::Group rot_group = rot_file.getGroup("expinfo_0");
    HighFive::DataSet rot_dataset = rot_group.getDataSet("goniometer_0");
    auto rotMatrix = rot_dataset.read<Eigen::Matrix3f>();

    rot_group = rot_file.getGroup("symmetryOps");
    auto n_elements = rot_group.getNumberObjects();
    std::vector<Eigen::Matrix3f> symm;
    for (size_t i = 0; i < n_elements; ++i) {
      rot_dataset = rot_group.getDataSet("op_" + std::to_string(i));
      symm.push_back(rot_dataset.read<Eigen::Matrix3f>());
    }

    rot_dataset = rot_file.getDataSet("ubmatrix");
    auto m_UB = rot_dataset.read<Eigen::Matrix3f>();

    std::vector<bool> skip_dets;
    rot_dataset = rot_file.getDataSet("skip_dets");
    rot_dataset.read(skip_dets);

    Eigen::Matrix3f m_W;
    m_W << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f;

    std::unordered_map<int32_t, size_t> fluxDetToIdx;
    std::unordered_map<int32_t, size_t> solidAngDetToIdx;

    std::vector<std::vector<double>> solidAngleWS;
    HighFive::File sa_file(GARNET_SA_NXS, HighFive::File::ReadOnly);
    HighFive::Group sa_group = sa_file.getGroup("mantid_workspace_1");
    HighFive::Group sa_group2 = sa_group.getGroup("workspace");
    HighFive::DataSet sa_dataset = sa_group2.getDataSet("values");
    std::vector<size_t> dims = sa_dataset.getDimensions();
    REQUIRE(dims.size() == 2);
    REQUIRE(dims[1] == 1);
    sa_dataset.read(solidAngleWS);

    sa_group2 = sa_group.getGroup("instrument");
    HighFive::Group sa_group3 = sa_group2.getGroup("detector");
    sa_dataset = sa_group3.getDataSet("detector_count");
    dims = sa_dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    std::vector<int> dc_data;
    sa_dataset.read(dc_data);

    std::vector<int> detIDs;
    sa_dataset = sa_group3.getDataSet("detector_list");
    dims = sa_dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    sa_dataset.read(detIDs);

    int detector{0};
    size_t idx{0};
    for (auto &value : dc_data) {
      for (int i = 0; i < value; ++i) {
        solidAngDetToIdx.emplace(detIDs[detector++], idx);
      }
      ++idx;
    }

    HighFive::File file(GARNET_FLUX_NXS, HighFive::File::ReadOnly);
    HighFive::Group group = file.getGroup("mantid_workspace_1");
    HighFive::Group group2 = group.getGroup("workspace");
    HighFive::DataSet dataset = group2.getDataSet("axis1");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    std::vector<double> read_data;
    dataset.read(read_data);
    reg integrFlux_x(read_data.size() - 1, read_data.front(), read_data.back(), "integrFlux_x");

    dataset = group2.getDataSet("values");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 2);
    REQUIRE(dims[0] == 75);
    std::vector<std::vector<double>> integrFlux_y{1};
    dataset.read(integrFlux_y);
    REQUIRE(integrFlux_y.size() == 75);

    group2 = group.getGroup("instrument");
    HighFive::Group group3 = group2.getGroup("detector");
    dataset = group3.getDataSet("detector_count");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 75);
    dataset.read(dc_data);

    dataset = group3.getDataSet("detector_list");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 307200);
    sa_dataset.read(detIDs);

    detector = 0;
    idx = 0;
    for (auto &value : dc_data) {
      for (int i = 0; i < value; ++i) {
        fluxDetToIdx.emplace(detIDs[detector++], idx);
      }
      ++idx;
    }

    group3 = group2.getGroup("physical_detectors");
    dataset = group3.getDataSet("number_of_detectors");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 1);
    size_t ndets;
    dataset.read(ndets);

    std::vector<float> lowValues, highValues;

    HighFive::File event_file(GARNET_EVENT_NXS, HighFive::File::ReadOnly);
    HighFive::Group event_group = event_file.getGroup("MDEventWorkspace");
    HighFive::Group event_group2 = event_group.getGroup("experiment0");
    HighFive::Group event_group3 = event_group2.getGroup("logs");
    HighFive::Group event_group4 = event_group3.getGroup("MDNorm_low");
    dataset = event_group4.getDataSet("value");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(lowValues);
    event_group4 = event_group3.getGroup("MDNorm_high");
    dataset = event_group4.getDataSet("value");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(highValues);

    event_group4 = event_group3.getGroup("gd_prtn_chrg");
    dataset = event_group4.getDataSet("value");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 1);
    double protonCharge;
    dataset.read(protonCharge);

    std::vector<float> thetaValues, phiValues;
    event_group3 = event_group2.getGroup("instrument");
    event_group4 = event_group3.getGroup("physical_detectors");
    dataset = event_group4.getDataSet("polar_angle");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(thetaValues);

#pragma omp parallel for
    for (auto &value : thetaValues)
      value *= boost::math::float_constants::degree;

    dataset = event_group4.getDataSet("azimuthal_angle");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(phiValues);

#pragma omp parallel for
    for (auto &value : phiValues)
      value *= boost::math::float_constants::degree;

    dataset = event_group4.getDataSet("detector_number");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(detIDs);

    // const char *EventHeaders[] = {"signal, errorSquared, center (each dim.)",
    //                              "signal, errorSquared, expInfoIndex, goniometerIndex, detectorId, center (each "
    //                              "dim.)"};
    // https://github.com/mantidproject/mantid/blob/c3ea43e4605f6898b84bd95c1196ccd8035364b1/Framework/DataObjects/src/BoxControllerNeXusIO.cpp#L27
    std::vector<std::vector<double>> events;
    event_group2 = event_group.getGroup("event_data");
    dataset = event_group2.getDataSet("event_data");
    dataset.read(events);

    auto signal = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                      std::get<1>(axes), std::get<2>(axes));

    std::vector<std::array<float, 4>> intersections;
    std::vector<float> xValues;
    std::vector<double> yValues;

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
    dims = norm_dataset.getDimensions();
    REQUIRE(dims.size() == 3);
    REQUIRE(dims[0] == 1);
    REQUIRE(dims[1] == 200);
    REQUIRE(dims[2] == 200);
    std::vector<std::vector<std::vector<double>>> data;
    norm_dataset.read(data);

    /*auto &data2d = data[0];
    double max_signal = *std::max_element(signal.begin(), signal.end());

    double ref_max{0.};
    for (size_t i = 0; i < dims[1]; ++i) {
      for (size_t j = 0; j < dims[2]; ++j) {
        REQUIRE_THAT(data2d[i][j], Catch::Matchers::WithinAbs(signal.at(j, i, 0), 5.e+05));
        ref_max = std::max(ref_max, data2d[i][j]);
      }
    }
    REQUIRE_THAT(max_signal, Catch::Matchers::WithinAbs(ref_max, 2.e+04));*/

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
