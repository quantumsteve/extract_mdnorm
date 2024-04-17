#include "calculateIntersections.h"
#include "validation_data_filepath.h"
#include "catch2/catch_all.hpp"

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

    double ndets;
    istrm >> ndets;

    std::vector<std::array<double, 4>> intersections;
    for(size_t i = 0; i < ndets; ++i)
    {
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
    }
  }
}
