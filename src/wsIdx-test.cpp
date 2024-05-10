#include "catch2/catch_all.hpp"
#include "validation_data_filepath.h"

#include <fstream>
#include <iostream>
#include <unordered_map>

TEST_CASE("getWSIdx") {
  SECTION("t0") {
    std::ifstream istrm(FLUXDET_TO_IDX_FILE);
    std::unordered_map<int32_t, size_t> fluxDetToIdx;
    std::unordered_map<int32_t, size_t> solidAngDetToIdx;
    size_t ndets, ndets_verify;
    istrm >> ndets;

    for (size_t i = 0; i < ndets; ++i) {
      int32_t first;
      size_t second;
      istrm >> first >> second;
      fluxDetToIdx.emplace(first, second);
    }

    std::ifstream sa_strm(SA_WS_FILE);
    double protonCharge, protonChargeBkgd;
    sa_strm >> protonCharge >> protonChargeBkgd;
    size_t sa_size;
    sa_strm >> sa_size;
    for (size_t i = 0; i < sa_size; ++i) {
      int32_t first;
      size_t second;
      sa_strm >> first >> second;
      solidAngDetToIdx.emplace(first, second);
    }

    std::vector<std::vector<double>> solidAngleWS;
    size_t sa_row, sa_col;
    sa_strm >> sa_row >> sa_col;
    for (size_t i = 0; i < sa_row; ++i) {
      REQUIRE(sa_col == 1);
      double value;
      sa_strm >> value;
      solidAngleWS.push_back({value});
    }

    istrm >> ndets_verify;
    REQUIRE(ndets == ndets_verify);

    bool haveSA{true};
    for (size_t i = 0; i < ndets; ++i) {
      size_t i_verify;
      int32_t detID;
      size_t wsIdx_verify;
      istrm >> i_verify >> detID >> wsIdx_verify;

      if (i != i_verify) {
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

      // Get solid angle for this contribution
      double solid = protonCharge;
      double bkgdSolid = protonChargeBkgd;
      if (haveSA) {
        double solid_angle_factor = solidAngleWS[solidAngDetToIdx.find(detID)->second][0];
        solid *= solid_angle_factor;
        bkgdSolid *= solid_angle_factor;
      }

      if (sa_strm.eof())
        continue;

      auto asdf = sa_strm.tellg();
      double solid_verify, bkgdSolid_verify;
      sa_strm >> i_verify >> solid_verify >> bkgdSolid_verify;
      if (i == i_verify) {
        REQUIRE_THAT(solid, Catch::Matchers::WithinAbs(solid_verify, 20));
        REQUIRE_THAT(bkgdSolid, Catch::Matchers::WithinAbs(bkgdSolid_verify, 0.0001));
      } else {
        sa_strm.seekg(asdf);
      }
    }
  }
}