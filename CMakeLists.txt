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

find_package(MPI REQUIRED)
find_package(Boost REQUIRED COMPONENTS MPI)
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
set( destination_name "validation_data_filepath.h" )
set( destination_path "${PROJECT_BINARY_DIR}/include/${destination_name}" )

set( SA_NXS "${CMAKE_SOURCE_DIR}/data/SingleCrystalDiffuseReduction_SA.nxs" )
set( FLUX_NXS "${CMAKE_SOURCE_DIR}/data/SingleCrystalDiffuseReduction_Flux.nxs" )
set( EVENT_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_Before_MDNorm.nxs" )
set( ROT_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_rotations.hdf5")
set( NORM_NXS "${CMAKE_SOURCE_DIR}/data/CORELLI_29782_After_MDNorm_symm.nxs" )

set( GARNET_SA_NXS "${CMAKE_SOURCE_DIR}/data/garnet/solid_angle_2p5-8.nxs")
set( GARNET_FLUX_NXS "${CMAKE_SOURCE_DIR}/data/garnet/flux_2p5-8.nxs")
set( GARNET_EVENT_NXS "${CMAKE_SOURCE_DIR}/data/garnet/CORELLI_BEFORE_MDNorm.nxs")
set( GARNET_ROT_NXS "${CMAKE_SOURCE_DIR}/data/garnet/CORELLI_extra_params.hdf5")

set( BENZIL_SA_NXS "${CMAKE_SOURCE_DIR}/data/benzil/SolidAngle20160720NoCC.nxs" )
set( BENZIL_FLUX_NXS "${CMAKE_SOURCE_DIR}/data/benzil/Spectrum20160720NoCC.nxs" )
set( BENZIL_EVENT_NXS_PREFIX "${CMAKE_SOURCE_DIR}/data/benzil/CORELLI_" )
set( BENZIL_EVENT_NXS_MIN "00" )
set( BENZIL_EVENT_NXS_MAX "35" )

set( BIXBYITE_SA_NXS "${CMAKE_SOURCE_DIR}/data/bixbyite/solidAngle.nxs" )
set( BIXBYITE_FLUX_NXS "${CMAKE_SOURCE_DIR}/data/bixbyite/spectra.nxs" )
set( BIXBYITE_EVENT_NXS_PREFIX "${CMAKE_SOURCE_DIR}/data/bixbyite/TOPAZ_" )
set( BIXBYITE_EVENT_NXS_MIN "40704" )
set( BIXBYITE_EVENT_NXS_MAX "40725" )

configure_file( ${source_path} ${destination_path} )

add_executable(calculateIntersections-test src/calculateIntersections.cpp src/calculateIntersections-test.cpp)
target_link_libraries(calculateIntersections-test HighFive Eigen3::Eigen Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_include_directories(calculateIntersections-test PRIVATE ${PROJECT_BINARY_DIR}/include)

add_executable(calcDiffractionIntersectionIntegral-test src/calculateIntersections.cpp
                                                        src/calcDiffractionIntersectionIntegral.cpp
                                                        src/calcDiffractionIntersectionIntegral-test.cpp
                                                        src/LoadFluxWorkspace.cpp
                                                        src/LoadSolidAngleWorkspace.cpp
                                                        src/LoadEventWorkspace.cpp
                                                        src/LoadExtrasWorkspace.cpp
                                                        )
target_link_libraries(calcDiffractionIntersectionIntegral-test HighFive Eigen3::Eigen Boost::boost Catch2::Catch2WithMain OpenMP::OpenMP_CXX)
target_include_directories(calcDiffractionIntersectionIntegral-test PRIVATE ${PROJECT_BINARY_DIR}/include)

add_executable(garnet_reduction_corelli src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/garnet_reduction_corelli.cpp
                                        src/LoadFluxWorkspace.cpp
                                        src/LoadSolidAngleWorkspace.cpp
                                        src/LoadEventWorkspace.cpp
                                        src/LoadExtrasWorkspace.cpp
                                        )
target_link_libraries(garnet_reduction_corelli HighFive Eigen3::Eigen Catch2::Catch2WithMain Boost::boost OpenMP::OpenMP_CXX)
target_include_directories(garnet_reduction_corelli PRIVATE ${PROJECT_BINARY_DIR}/include)

add_executable(benzil_corelli src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/benzil_corelli.cpp
                                        src/LoadFluxWorkspace.cpp
                                        src/LoadSolidAngleWorkspace.cpp
                                        src/LoadEventWorkspace.cpp
                                        src/LoadExtrasWorkspace.cpp
                                        )
target_link_libraries(benzil_corelli HighFive Eigen3::Eigen Catch2::Catch2WithMain Boost::boost Boost::mpi ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(benzil_corelli PRIVATE ${PROJECT_BINARY_DIR}/include ${MPI_CXX_INCLUDE_PATH})

add_executable(bixbyite_topaz src/calculateIntersections.cpp
                                        src/calcDiffractionIntersectionIntegral.cpp
                                        src/bixbyite_topaz.cpp 
                                        src/LoadFluxWorkspace.cpp
                                        src/LoadSolidAngleWorkspace.cpp
                                        src/LoadEventWorkspace.cpp
                                        src/LoadExtrasWorkspace.cpp
                                        )

target_link_libraries(bixbyite_topaz HighFive Eigen3::Eigen Catch2::Catch2WithMain Boost::boost Boost::mpi ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)
target_include_directories(bixbyite_topaz PRIVATE ${PROJECT_BINARY_DIR}/include ${MPI_CXX_INCLUDE_PATH} )

include( CTest )

enable_testing()

add_test( NAME calculateIntersections COMMAND calculateIntersections-test)
add_test( NAME calcDiffractionIntersectionIntegral COMMAND calcDiffractionIntersectionIntegral-test )
