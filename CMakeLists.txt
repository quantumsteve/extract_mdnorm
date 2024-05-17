cmake_minimum_required( VERSION 3.22 )

project( mdnorm CXX)

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

# Set a default build type if none was specified
# https://blog.kitware.com/cmake-and-the-default-build-type/
set (default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE
       STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
else ()
  message (STATUS "CMAKE_BUILD_TYPE is ${CMAKE_BUILD_TYPE}")
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

include( CMake/catch.cmake )

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED)
find_package(OpenMP REQUIRED)
set(HIGHFIVE_GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git)
set(HIGHFIVE_GIT_TAG d79b5acc758cc9c5e1733afc88d3b3ca66316150)
set(HIGHFIVE_UNIT_TESTS OFF)
set(HIGHFIVE_USE_BOOST OFF)
set(HIGHFIVE_USE_EIGEN ON)

include(FetchContent)
FetchContent_Declare(HighFive
  GIT_REPOSITORY ${HIGHFIVE_GIT_REPOSITORY}
  GIT_TAG ${HIGHFIVE_GIT_TAG}
)
FetchContent_MakeAvailable(HighFive)

# header containing all the filepaths as preprocessor defines
set( source_name "src/validation_data_filepath.h.in" )
set( source_path "${CMAKE_SOURCE_DIR}/${source_name}" )
set( destination_name "src/validation_data_filepath.h" )
set( destination_path "${CMAKE_SOURCE_DIR}/${destination_name}" )
set( CALC_INTERSECTIONS_FILE "${CMAKE_SOURCE_DIR}/data/corelli_demo.txt" )
set( FLUXDET_TO_IDX_FILE "${CMAKE_SOURCE_DIR}/data/fluxDetToIdx.txt" )
set( SA_WS_FILE "${CMAKE_SOURCE_DIR}/data/solidAngDetToIdx.txt" )
set( FLUX_NXS "${CMAKE_SOURCE_DIR}/data/SingleCrystalDiffuseReduction_Flux.nxs" )
set( SA_NXS "${CMAKE_SOURCE_DIR}/data/SingleCrystalDiffuseReduction_SA.nxs" )
set( EVENT_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_Before_MDNorm.nxs" )
set( NORM_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_After_MDNorm_symm.nxs" )
set( USE_DETS_FILE "${CMAKE_SOURCE_DIR}/data/use_det.txt")
set( ROT_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_rotations.hdf5")
set( GARNET_EVENT_NXS "${CMAKE_SOURCE_DIR}/data/garnet/CORELLI_BEFORE_MDNorm.nxs")
set( GARNET_SA_NXS "${CMAKE_SOURCE_DIR}/data/garnet/solid_angle_2p5-8.nxs")
set( GARNET_FLUX_NXS "${CMAKE_SOURCE_DIR}/data/garnet/flux_2p5-8.nxs")
set( GARNET_ROT_NXS "${CMAKE_SOURCE_DIR}/data/garnet/CORELLI_extra_params.hdf5")

configure_file( ${source_path} ${destination_path} )

add_executable(calculateIntersections-test src/calculateIntersections.cpp src/calculateIntersections-test.cpp)
target_link_libraries(calculateIntersections-test Eigen3::Eigen Catch2::Catch2WithMain)

add_executable(wsIdx-test src/wsIdx-test.cpp)
target_link_libraries(wsIdx-test Catch2::Catch2WithMain)

add_executable(calcDiffractionIntersectionIntegral-test src/calculateIntersections.cpp
                                                        src/calcDiffractionIntersectionIntegral.cpp
                                                        src/calcDiffractionIntersectionIntegral-test.cpp )

target_link_libraries(calcDiffractionIntersectionIntegral-test HighFive Eigen3::Eigen Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

add_executable(garnet_reduction_corelli src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/garnet_reduction_corelli.cpp )

target_link_libraries(garnet_reduction_corelli HighFive Eigen3::Eigen Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

add_executable(benzil_corelli src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/benzil_corelli.cpp )

target_link_libraries(benzil_corelli HighFive Eigen3::Eigen Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

add_executable(bixbyite_topaz src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/bixbyite_topaz.cpp )

target_link_libraries(bixbyite_topaz HighFive Eigen3::Eigen Catch2::Catch2WithMain OpenMP::OpenMP_CXX)

include( CTest )

enable_testing()

add_test( NAME calculateIntersections COMMAND calculateIntersections-test )
add_test( NAME calcDiffractionIntersectionIntegral COMMAND calcDiffractionIntersectionIntegral-test )
add_test( NAME wsIdx COMMAND wsIdx-test )

