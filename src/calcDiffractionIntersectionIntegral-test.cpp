#include "calculateIntersections.h"
#include "calcDiffractionIntersectionIntegral.h"
#include "calcSingleDetectorNorm.h"

#include "validation_data_filepath.h"
#include "catch2/catch_all.hpp"

#include <highfive/highfive.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

TEST_CASE("calculateIntersections") {
  SECTION("t0") {
    std::vector<double> hX(201), kX(201);
    for(int i = 0; i < 201; ++i)
    {
      hX[i] = kX[i] =  0.1 * static_cast<double>(i) - 10.;
    }
    std::vector<double> lX{-0.1, 0.1};

    MDNorm doctest(hX, kX, lX);
    
    std::ifstream istrm(CALC_INTERSECTIONS_FILE);

    Eigen::Matrix3d transform;
    istrm >> transform(0,0) >> transform(0,1) >> transform(0,2) 
	        >> transform(1,0) >> transform(1,1) >> transform(1,2)
	        >> transform(2,0) >> transform(2,1) >> transform(2,2);

    size_t ndets;
    istrm >> ndets;

    std::ifstream flux_strm(FLUXDET_TO_IDX_FILE);
    std::unordered_map<int32_t, size_t> fluxDetToIdx;
    std::unordered_map<int32_t, size_t> solidAngDetToIdx;
    size_t ndets_verify;
    flux_strm >> ndets_verify;

    REQUIRE(ndets == ndets_verify);

    for(size_t i = 0; i < ndets;++i)
    {
        int32_t first;
        size_t second;
        flux_strm >> first >> second;
        fluxDetToIdx.emplace(first, second);
    }

    std::ifstream sa_strm(SA_WS_FILE);
    double protonCharge, protonChargeBkgd;
    sa_strm >> protonCharge >> protonChargeBkgd;
    size_t sa_size;
    sa_strm >> sa_size;
    for(size_t i = 0; i < sa_size;++i)
    {
        int32_t first;
        size_t second;
        sa_strm >> first >> second;
        solidAngDetToIdx.emplace(first, second);
    }

    std::vector<std::vector<double>> solidAngleWS; 
    size_t sa_row, sa_col;
    sa_strm >> sa_row >> sa_col;
    REQUIRE(sa_col == 1);
    for(size_t i = 0; i < sa_row; ++i)
    {
      double value;
      sa_strm >> value;
      // solidAngleWS.push_back({value});
    }

    HighFive::File sa_file(SA_NXS, HighFive::File::ReadOnly);
    HighFive::Group sa_group = sa_file.getGroup("mantid_workspace_1");
    HighFive::Group sa_group2 = sa_group.getGroup("workspace");
    HighFive::DataSet sa_dataset = sa_group2.getDataSet("values");
    std::vector<size_t> dims = sa_dataset.getDimensions();
    REQUIRE(dims[1] == 1); 
    std::vector<double> read_data;
    sa_dataset.read(read_data);
    
    flux_strm >> ndets_verify;
    REQUIRE(ndets == ndets_verify); 

    for(const double value: read_data)
      solidAngleWS.push_back({value});

    bool haveSA{true};

    HighFive::File file(FLUX_NXS, HighFive::File::ReadOnly);
    HighFive::Group group = file.getGroup("mantid_workspace_1");
    HighFive::Group group2 = group.getGroup("workspace");
    HighFive::DataSet dataset = group2.getDataSet("axis1");
    dims = dataset.getDimensions();
    dataset.read(read_data);
    REQUIRE(dims.size() == 1);
    std::vector<std::vector<double>> integrFlux_x{1}, integrFlux_y{1};
    for(size_t j = 0; j < dims[0]; ++j)
      integrFlux_x[0].push_back(read_data[j]);

    dataset = group2.getDataSet("values");
    dims = dataset.getDimensions();
    dataset.read(read_data);

    REQUIRE(dims.size() == 2);
    REQUIRE(dims[0] == 1);
    for(size_t j = 0; j < dims[1]; ++j)
      integrFlux_y[0].push_back(read_data[j]);

    std::vector<bool> use_dets;
    std::ifstream dets_strm(USE_DETS_FILE);
    for(size_t i = 0;i < ndets;++i) {
      REQUIRE(!dets_strm.eof());
      bool value{false};
      dets_strm >> value;
      REQUIRE(value == true);
      use_dets.push_back(value);
    }

    std::vector<double> lowValues, highValues;

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

    std::vector<double> thetaValues, phiValues;
    event_group3 = event_group2.getGroup("instrument");
    event_group4 = event_group3.getGroup("physical_detectors");
    dataset = event_group4.getDataSet("polar_angle");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(thetaValues);

    for(auto &value : thetaValues)
      value = value * M_PI/180.;

    dataset = event_group4.getDataSet("azimuthal_angle");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(phiValues);

    for(auto &value: phiValues)
      value = value * M_PI/180.;

    std::vector<int> detIDs;
    dataset = event_group4.getDataSet("detector_number");
    dims = dataset.getDimensions();
    REQUIRE(dims.size() == 1);
    REQUIRE(dims[0] == 372736);
    dataset.read(detIDs);

    std::vector<std::atomic<double>> signalArray(200*200);

    std::vector<std::array<double, 4>> intersections;
    std::vector<double> xValues, yValues;
    std::vector<float> pos, posNew;
    const size_t vmdDims =  3;

    #pragma omp parallel for private(intersections, xValues, yValues, pos, posNew)
    for(size_t i = 0; i < ndets; ++i)
    {
      if(!use_dets[i])
        continue;

      int32_t detID = detIDs[i];
      // get the flux spectrum number: this is for diffraction only!
      size_t wsIdx = 0;
      if (auto index = fluxDetToIdx.find(detID); index != fluxDetToIdx.end())
        wsIdx = index->second;
      else // masked detector in flux, but not in input workspace
        continue;

      doctest.calculateIntersections(intersections, thetaValues[i], phiValues[i], transform, lowValues[i], highValues[i]);

      if(intersections.empty())
        continue;

      // Get solid angle for this contribution
      double solid = protonCharge;
      if (haveSA) {
        double solid_angle_factor = solidAngleWS[solidAngDetToIdx.find(detID)->second][0]; 
        solid *= solid_angle_factor;
      }

      calcDiffractionIntersectionIntegral(intersections, xValues, yValues, integrFlux_x, integrFlux_y, wsIdx);

      pos.resize(vmdDims);
      posNew.resize(vmdDims);

      calcSingleDetectorNorm(intersections, solid, yValues, vmdDims, pos, posNew, signalArray);
    }

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

    auto &data2d = data[0];

    double max_signal = *std::max_element(signalArray.begin(),signalArray.end());

    double ref_max{0.};
    for(size_t i = 0; i < dims[1];++i) {
      for(size_t j = 0; j < dims[2];++j) {
        REQUIRE_THAT(data2d[i][j],Catch::Matchers::WithinAbs(signalArray[i*dims[1] + j],2.e+04));
        ref_max = std::max(ref_max,data2d[i][j]);
      }
    }
    REQUIRE_THAT(max_signal,Catch::Matchers::WithinAbs(ref_max,2.e+04));

    //std::ofstream out_strm("meow.txt");
    //for(size_t i = 0; i < signalArray.size();++i)
    //  out_strm << signalArray[i] << '\n'; 
  }
}