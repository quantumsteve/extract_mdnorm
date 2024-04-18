cmake_minimum_required( VERSION 3.22 )

project( mdnorm CXX)

set( CMAKE_CXX_STANDARD 17 )
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
# header containing all the filepaths as preprocessor defines
set( source_name "src/validation_data_filepath.h.in" )
set( source_path "${CMAKE_SOURCE_DIR}/${source_name}" )
set( destination_name "src/validation_data_filepath.h" )
set( destination_path "${CMAKE_SOURCE_DIR}/${destination_name}" )
set( CALC_INTERSECTIONS_FILE "${CMAKE_SOURCE_DIR}/data/corelli_demo.txt")
configure_file( ${source_path} ${destination_path} )

add_executable(calculateIntersections-test src/calculateIntersections.cpp src/calculateIntersections-test.cpp)
target_link_libraries(calculateIntersections-test Eigen3::Eigen Catch2::Catch2WithMain)

include( CTest )

enable_testing()

add_test( NAME calculateIntersections COMMAND calculateIntersections-test )

