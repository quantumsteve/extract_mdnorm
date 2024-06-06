#include "LoadEventWorkspace.h"
#include "LoadExtrasWorkspace.h"
#include "LoadFluxWorkspace.h"
#include "LoadSolidAngleWorkspace.h"
#include "calcDiffractionIntersectionIntegral.h"
#include "calculateIntersections.h"

#include "validation_data_filepath.h"

#include <boost/histogram.hpp>
#include <boost/histogram/serialization.hpp>
#include <boost/mpi.hpp>
#include <highfive/highfive.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <tuple>
#include <vector>

using histogram_type = boost::histogram::histogram<
    std::tuple<boost::histogram::axis::regular<float>, boost::histogram::axis::regular<float>,
               boost::histogram::axis::regular<float>>,
    boost::histogram::dense_storage<boost::histogram::accumulators::thread_safe<double>>>;

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
  std::tuple<reg, reg, reg> axes{reg(601, -16.0, 16.0, "x"), reg(601, -16.0, 16.0, "y"), reg(601, -16.0, 16.0, "z")};

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

  auto signal = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                    std::get<1>(axes), std::get<2>(axes));

  auto h = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes), std::get<1>(axes),
                               std::get<2>(axes));

  Eigen::Matrix3f m_W;
  m_W << 1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f;

  LoadSolidAngleWorkspace solidAngle(BIXBYITE_SA_NXS);
  const std::unordered_map<int32_t, size_t> solidAngDetToIdx = solidAngle.getSolidAngDetToIdx();
  const std::vector<std::vector<double>> solidAngleWS = solidAngle.getSolidAngleValues();

  LoadFluxWorkspace flux(BIXBYITE_FLUX_NXS);
  const reg integrFlux_x = flux.getFluxAxis();
  const std::vector<std::vector<double>> integrFlux_y = flux.getFluxValues();
  const std::unordered_map<int32_t, size_t> fluxDetToIdx = flux.getFluxDetToIdx();
  const size_t ndets = flux.getNDets();

  std::string rot_filename = std::string(BIXBYITE_EVENT_NXS_PREFIX).append("40704_extra_params.hdf5");
  LoadExtrasWorkspace extras(rot_filename);
  const std::vector<Eigen::Matrix3f> symm = extras.getSymmMatrices();
  const Eigen::Matrix3f m_UB = extras.getUBMatrix();
  const std::vector<bool> skip_dets = extras.getSkipDets();

  auto event_filename = std::string(BIXBYITE_EVENT_NXS_PREFIX).append("40704_BEFORE_MDNorm.nxs");

  LoadEventWorkspace eventWS(event_filename);
  const std::vector<float> lowValues = eventWS.getLowValues();
  const std::vector<float> highValues = eventWS.getHighValues();
  const std::vector<float> thetaValues = eventWS.getThetaValues();
  const std::vector<float> phiValues = eventWS.getPhiValues();
  const std::vector<int> detIDs = eventWS.getDetIDs();

  std::vector<Eigen::Matrix3f> transforms2;
  for (const Eigen::Matrix3f &op : symm) {
    Eigen::Matrix3f transform = m_UB * op * m_W;
    transforms2.push_back(transform.inverse());
  }

  std::vector<int> idx;
  std::vector<float> momentum;
  std::vector<Eigen::Vector3f> intersections;
  std::vector<float> xValues;
  std::vector<double> yValues;
  std::vector<Eigen::Matrix3f> transforms;
  std::vector<std::array<double, 8>> events;

  int rank = world.rank();
  int N = BIXBYITE_EVENT_NXS_MAX - BIXBYITE_EVENT_NXS_MIN + 1;

  int count = N / world.size();
  int remainder = N % world.size();
  int start, stop;
  if (rank < remainder) {
    // The first 'remainder' ranks get 'count + 1' tasks each
    start = rank * (count + 1);
    stop = start + count;
  } else {
    // The remaining 'size - remainder' ranks get 'count' task each
    start = rank * count + remainder;
    stop = start + (count - 1);
  }
  std::cout << "rank: " << rank << " " << N << " " << start << " " << stop << std::endl;
  for (int file_num = BIXBYITE_EVENT_NXS_MIN + start; file_num <= BIXBYITE_EVENT_NXS_MIN + stop; ++file_num) {
    auto rot_filename_changes =
        std::string(BIXBYITE_EVENT_NXS_PREFIX).append(std::to_string(file_num)).append("_extra_params.hdf5");
    LoadExtrasWorkspace extras_changes(rot_filename_changes);
    auto rotMatrix = extras_changes.getRotMatrix();

    transforms.clear();
    for (const Eigen::Matrix3f &op : symm) {
      Eigen::Matrix3f transform = rotMatrix * m_UB * op * m_W;
      transforms.push_back(transform.inverse());
    }

    auto event_filename_changes =
        std::string(BIXBYITE_EVENT_NXS_PREFIX).append(std::to_string(file_num)).append("_BEFORE_MDNorm.nxs");
    LoadEventWorkspace eventWS_changes(event_filename_changes);
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
        {
          continue;
        }

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
    std::cout << "rank: " << rank << " MDNorm time: " << duration_total << "s\n";

    /*HighFive::File norm_file(BIXBYITE_EVENT_NXS_PREFIX+"_0_norm.hdf5",
    HighFive::File::ReadOnly); HighFive::Group norm_group = norm_file.getGroup("MDHistoWorkspace"); HighFive::Group
    norm_group2 = norm_group.getGroup("data"); HighFive::DataSet norm_dataset = norm_group2.getDataSet("signal"); dims
    = norm_dataset.getDimensions(); REQUIRE(dims.size() == 3); REQUIRE(dims[0] == 1); REQUIRE(dims[1] == 603);
    REQUIRE(dims[2] == 603);
    std::vector<std::vector<std::vector<double>>> data;
    norm_dataset.read(data);

    auto &data2d = data[0];
    double max_signal = *std::max_element(signal.begin(), signal.end());

    double ref_max{0.};
    for (size_t i = 0; i < dims[1]; ++i) {
      for (size_t j = 0; j < dims[2]; ++j) {
        //REQUIRE_THAT(data2d[i][j], Catch::Matchers::WithinAbs(signal.at(j, i, 0), 5.e+05));
        ref_max = std::max(ref_max, data2d[i][j]);
      }
    }
    //REQUIRE_THAT(max_signal, Catch::Matchers::WithinAbs(ref_max, 2.e+04));*/

    eventWS_changes.updateEvents(events);

    startt = std::chrono::high_resolution_clock::now();
#pragma omp parallel for collapse(2)
    for (const Eigen::Matrix3f &op : transforms2) {
      for (auto &val : events) {
        Eigen::Vector3f v(val[5], val[6], val[7]);
        v = op * v;
        h(v[0], v[1], v[2], weight(val[0]));
      }
    }
    stopt = std::chrono::high_resolution_clock::now();
    duration_total = std::chrono::duration<double, std::chrono::seconds::period>(stopt - startt).count();
    std::cout << "rank: " << rank << " BinMD time: " << duration_total << "s\n";

    /*HighFive::File data_file(BIXBYITE_EVENT_NXS_PREFIX+"0_data.hdf5",
    HighFive::File::ReadOnly); HighFive::Group data_group = data_file.getGroup("MDHistoWorkspace"); HighFive::Group
    data_group2 = data_group.getGroup("data"); HighFive::DataSet data_dataset = data_group2.getDataSet("signal"); dims
    = data_dataset.getDimensions(); REQUIRE(dims.size() == 3); REQUIRE(dims[0] == 1); REQUIRE(dims[1] == 603);
    REQUIRE(dims[2] == 603);
    norm_dataset.read(data);

    data2d = data[0];
    max_signal = *std::max_element(h.begin(), h.end());

    ref_max = 0.;
    for (size_t i = 0; i < dims[1]; ++i) {
      for (size_t j = 0; j < dims[2]; ++j) {
        //REQUIRE_THAT(data2d[i][j], Catch::Matchers::WithinAbs(h.at(j, i, 0), 5.e+05));
        ref_max = std::max(ref_max, data2d[i][j]);
      }
    }
    //REQUIRE_THAT(max_signal, Catch::Matchers::WithinAbs(ref_max, 2.e+04));
    std::cout << max_signal << " " <<  ref_max << std::endl;*/
  }

  events.clear();
  events.shrink_to_fit();

  histogram_type numerator, denominator;

  if (world.rank() == 0) {
    numerator = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                    std::get<1>(axes), std::get<2>(axes));

    denominator = make_histogram_with(dense_storage<accumulators::thread_safe<double>>(), std::get<0>(axes),
                                      std::get<1>(axes), std::get<2>(axes));
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
      num.push_back(*x);

    std::vector<double> denom;
    for (auto &&x : indexed(denominator))
      denom.push_back(*x);

    std::ofstream out_strm("meow.txt");
    for (size_t i = 0; i < num.size(); ++i)
      out_strm << num[i] / denom[i] << '\n';
  }
  return 0;
}
