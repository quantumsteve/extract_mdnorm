#include "calculateIntersections.h"

#include "catch2/catch_all.hpp"
#include "validation_data_filepath.h"

#include <boost/histogram.hpp>
#include <boost/math/constants/constants.hpp>
#include <highfive/eigen.hpp>
#include <highfive/highfive.hpp>

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

    HighFive::File rot_file(ROT_NXS, HighFive::File::ReadOnly);
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
    m_W << 1.f, 1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 0.f, 1.f;

    std::unordered_map<int32_t, size_t> fluxDetToIdx;

    HighFive::File file(FLUX_NXS, HighFive::File::ReadOnly);
    HighFive::Group group = file.getGroup("mantid_workspace_1");
    HighFive::Group group2 = group.getGroup("workspace");
    HighFive::DataSet dataset = group2.getDataSet("axis1");
    auto dims = dataset.getDimensions();
    std::vector<float> read_data_f;
    dataset.read(read_data_f);
    REQUIRE(dims.size() == 1);
    const reg integrFlux_x(read_data_f.size() - 1, read_data_f.front(), read_data_f.back(), "integrFlux_x");
    REQUIRE_THAT(integrFlux_x.bin(0).width(), Catch::Matchers::WithinAbs(read_data_f[1] - read_data_f[0], 1e-4));

    std::vector<double> read_data;
    dataset = group2.getDataSet("values");
    dims = dataset.getDimensions();
    dataset.read(read_data);

    REQUIRE(dims.size() == 2);
    REQUIRE(dims[0] == 1);
    std::vector<std::vector<double>> integrFlux_y{1};
    for (size_t j = 0; j < dims[1]; ++j)
      integrFlux_y[0].push_back(read_data[j]);

    std::vector<int> dc_data;
    group2 = group.getGroup("instrument");
    HighFive::Group group3 = group2.getGroup("detector");
    dataset = group3.getDataSet("detector_count");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 1);
    dataset.read(dc_data);

    std::vector<int> detIDs;
    dataset = group3.getDataSet("detector_list");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(detIDs);

    int detector = 0;
    int idx = 0;
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

    HighFive::File event_file(EVENT_NXS, HighFive::File::ReadOnly);
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

    for (auto &value : thetaValues)
      value *= boost::math::float_constants::degree;

    dataset = event_group4.getDataSet("azimuthal_angle");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(phiValues);

    for (auto &value : phiValues)
      value *= boost::math::float_constants::degree;

    dataset = event_group4.getDataSet("detector_number");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(detIDs);

    auto signal = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                      std::get<1>(axes), std::get<2>(axes));

    std::vector<Eigen::Matrix3f> transforms;
    for (const Eigen::Matrix3f &op : symm) {
      Eigen::Matrix3f transform = rotMatrix * m_UB * op * m_W;
      transforms.push_back(transform.inverse());
    }

    std::vector<std::array<float, 4>> intersections, intersections_new;
    for (const Eigen::Matrix3f &op : transforms) {
      for (size_t i = 0; i < ndets; ++i) {
        if (skip_dets[i])
          continue;

        int32_t detID = detIDs[i];
        // get the flux spectrum number: this is for diffraction only!
        if (auto index = fluxDetToIdx.find(detID); index == fluxDetToIdx.end())
          continue;

        doctest.calculateIntersections(intersections, thetaValues[i], phiValues[i], op, lowValues[i], highValues[i]);

        doctest.calculateIntersections(signal, intersections_new, thetaValues[i], phiValues[i], op, lowValues[i],
                                       highValues[i]);
        REQUIRE(intersections.size() == intersections_new.size());
        for (size_t j = 0; j < intersections.size(); ++j) {
          const auto &old_v = intersections[j];
          const auto &new_v = intersections_new[j];
          for (int k = 0; k < 4; ++k)
            REQUIRE_THAT(new_v[k], Catch::Matchers::WithinAbs(old_v[k], 0.0001));
        }
      }
    }
  }
}
