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

    std::vector<std::atomic<double>> signalArray(200*200);

    std::vector<std::array<double, 4>> intersections;
    std::vector<double> xValues, yValues;
    std::vector<float> pos, posNew;
    const size_t vmdDims =  3;
    for(size_t i = 0; i < ndets; ++i)
    {
      size_t i_verify;
      int32_t detID;
      size_t wsIdx_verify;
      flux_strm >> i_verify >> detID >> wsIdx_verify;
      if(i != i_verify)
      {
        REQUIRE(i < i_verify);
        i = i_verify;
      }

      // get the flux spectrum number: this is for diffraction only!
      size_t wsIdx = 0;
      if (auto index = fluxDetToIdx.find(detID); index != fluxDetToIdx.end())
        wsIdx = index->second;
      else // masked detector in flux, but not in input workspace
        continue;
      REQUIRE(wsIdx == wsIdx_verify);

      size_t i_f, num_intersections;
      double theta, phi, lowvalue, highvalue;     
      istrm >> i_f >> theta >> phi >> lowvalue >> highvalue;
      if(i != i_f)
        i = i_f;
      doctest.calculateIntersections(intersections, theta, phi, transform, lowvalue, highvalue);
      istrm >> num_intersections;
      REQUIRE(intersections.size() == num_intersections);
      for(auto & elem: intersections) {
        std::array<double, 4> values;
        istrm >> values[0] >> values[1] >> values[2] >> values[3];
        for(int j = 0; j < 4; ++j)
          REQUIRE_THAT(elem[j], Catch::Matchers::WithinAbs(values[j], 0.0001));
      }

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