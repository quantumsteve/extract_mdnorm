#include "histogram.h"
#include "mdnorm.h"
#include "parameters.h"
#include "validation_data_filepath.h"

#include <boost/histogram/serialization.hpp>
#include <boost/mpi.hpp>
#include <highfive/highfive.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

namespace boost {
namespace mpi {
template <> struct is_commutative<std::plus<histogram_type>, histogram_type> : mpl::true_ {};
} // namespace mpi
} // namespace boost

int main(int argc, char *argv[]) {

  namespace mpi = boost::mpi;
  namespace mt = mpi::threading;
  mpi::environment env(argc, argv, mt::funneled);
  if (env.thread_level() < mt::funneled) {
    env.abort(-1);
  }
  mpi::communicator world;

  using namespace boost::histogram;
  using reg = axis::regular<float>;

  int rank = world.rank();
  int N = 603;
  int count = N / world.size();
  int remainder = N % world.size();
  int binStart{0};
  int binStop{0};
  float valStart{0};
  float valStop{0};
  if (rank < remainder) {
    // The first 'remainder' ranks get 'count + 1' tasks each
    binStart = rank * (count + 1);
    binStop = binStart + count;
  } else {
    // The remaining 'size - remainder' ranks get 'count' task each
    binStart = rank * count + remainder;
    binStop = binStart + (count - 1);
  }
  std::cout << rank << " " << binStart << " " << binStop << std::endl;
  {
    reg tmp(N, -16., 16.);
    valStart = tmp.bin(binStart).lower();
    valStop = tmp.bin(binStop).upper();
  }
  std::cout << rank << " " << binStop - binStart + 1 << " " << valStart << " " << valStop << std::endl;

  parameters params;
  params.axes = std::tuple<reg, reg, reg>{reg(binStop - binStart + 1, valStart, valStop, "x"),
                                          reg(603, -16.0, 16.0, "y"), reg(1, -0.1, 0.1, "z")};
  params.W << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f;
  params.solidAngleFilename = BIXBYITE_SA_NXS;
  params.fluxFilename = BIXBYITE_FLUX_NXS;
  params.eventPrefix = BIXBYITE_EVENT_NXS_PREFIX;
  params.eventMin = BIXBYITE_EVENT_NXS_MIN;
  params.eventMax = BIXBYITE_EVENT_NXS_MAX;
  params.eventStart = 0;
  params.eventStop = params.eventMax - params.eventMin;
  /*int rank = world.rank();
  int N = params.eventMax - params.eventMin + 1;
  int count = N / world.size();
  int remainder = N % world.size();
  if (rank < remainder) {
    // The first 'remainder' ranks get 'count + 1' tasks each
    params.eventStart = rank * (count + 1);
    params.eventStop = params.eventStart + count;
  } else {
    // The remaining 'size - remainder' ranks get 'count' task each
    params.eventStart = rank * count + remainder;
    params.eventStop = params.eventStart + (count - 1);
  }*/

  histogram_type h, signal;
  mdnorm(params, h, signal);

  std::vector<double> num;
  for (auto &&x : indexed(h))
    num.push_back(static_cast<double>(*x));

  std::vector<double> denom;
  for (auto &&x : indexed(signal))
    denom.push_back(static_cast<double>(*x));

  std::ofstream out_strm("meow_" + std::to_string(world.rank()) + ".txt");
  for (size_t i = 0; i < num.size(); ++i)
    out_strm << num[i] / denom[i] << '\n';

  return 0;
}
