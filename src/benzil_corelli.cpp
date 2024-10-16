#include "LoadEventWorkspace.h"
#include "LoadExtrasWorkspace.h"
#include "LoadFluxWorkspace.h"
#include "LoadSolidAngleWorkspace.h"
#include "calcDiffractionIntersectionIntegral.h"
#include "calculateIntersections.h"
#include "histogram.h"
#include "mdnorm.h"
#include "parameters.h"
#include "validation_data_filepath.h"

#include <Eigen/Core>
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

  parameters params;
  params.axes = std::tuple<reg, reg, reg>{reg(603, -7.5375, 7.5375, "x"), reg(603, -13.16524, 13.16524, "y"),
                                          reg(1, -0.5, 0.5, "z")};
  params.W << 1.f, 1.f, 0.f, 1.f, -1.f, 0.f, 0.f, 0.f, 1.f;
  params.solidAngleFilename = BENZIL_SA_NXS;
  params.fluxFilename = BENZIL_FLUX_NXS;
  params.eventPrefix = BENZIL_EVENT_NXS_PREFIX;
  params.eventMin = BENZIL_EVENT_NXS_MIN;
  params.eventMax = BENZIL_EVENT_NXS_MAX;

  int rank = world.rank();
  int N = BENZIL_EVENT_NXS_MAX - BENZIL_EVENT_NXS_MIN + 1;
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
  }

  histogram_type h, signal;
  mdnorm(params, h, signal);
  histogram_type numerator, denominator;

  if (world.rank() == 0) {
    numerator = make_histogram_with(dense_storage<accumulator_type>(), std::get<0>(params.axes),
                                    std::get<1>(params.axes), std::get<2>(params.axes));

    denominator = make_histogram_with(dense_storage<accumulator_type>(), std::get<0>(params.axes),
                                      std::get<1>(params.axes), std::get<2>(params.axes));
  }

  auto startt = std::chrono::high_resolution_clock::now();
  reduce(world, h, numerator, std::plus<histogram_type>(), 0);
  reduce(world, signal, denominator, std::plus<histogram_type>(), 0);
  auto stopt = std::chrono::high_resolution_clock::now();
  auto duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
  std::cout << "rank: " << rank << " reduce time: " << duration_total << "s\n";

  if (world.rank() == 0) {
    std::vector<double> num;
    for (auto &&x : indexed(numerator))
      num.push_back(static_cast<double>(*x));

    std::vector<double> denom;
    for (auto &&x : indexed(denominator))
      denom.push_back(static_cast<double>(*x));

    std::ofstream out_strm("meow.txt");
    for (size_t i = 0; i < num.size(); ++i)
      out_strm << num[i] / denom[i] << '\n';
  }
  return 0;
}
