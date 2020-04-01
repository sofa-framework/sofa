# - Try to find CGAL
# Once done this will define
#
#  CGAL_FOUND        - system has CGAL
#  CGAL_INCLUDE_DIRS - include directories for CGAL
#  CGAL_LIBRARIES    - libraries for CGAL
#  CGAL_DEFINITIONS  - compiler flags for CGAL

#=============================================================================
# Copyright (C) 2010-2011 Anders Logg, Johannes Ring and Garth N. Wells
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

message(STATUS "Checking for package 'CGAL'")

# Blank out CGAL_FIND_VERSION temporarily or else find_package(CGAL ...)
# (below) will fail.
set(CGAL_FIND_VERSION_TMP ${CGAL_FIND_VERSION})
set(CGAL_FIND_VERSION "")

# Call CGAL supplied CMake script
find_package(CGAL
  HINTS
  ${CGAL_DIR}
  $ENV{CGAL_DIR}
   /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake
  PATH_SUFFIXES lib cmake/modules lib/cmake lib/CGAL)

# Restore CGAL_FIND_VERSION
set(CGAL_FIND_VERSION ${CGAL_FIND_VERSION_TMP})

if (CGAL_FIND_VERSION)
  # Check if version found is >= required version
  if (NOT "${CGAL_VERSION}" VERSION_LESS "${CGAL_FIND_VERSION}")
    set(CGAL_VERSION_OK TRUE)
  endif()
else()
  # No specific version of CGAL is requested
  set(CGAL_VERSION_OK TRUE)
endif()

# Add flag to fix bug in CGAL 4.1 for Intel compilers. See
# https://sympa.inria.fr/sympa/arc/cgal-discuss/2013-01/msg00011.html
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  if ("${CGAL_VERSION}" VERSION_GREATER "4.0.2")
    set(CGAL_DEFINITIONS "-DCGAL_CFG_NO_STATEMENT_EXPRESSIONS")
  endif()
endif()

# Set variables
set(CGAL_INCLUDE_DIRS ${CGAL_INCLUDE_DIRS} ${CGAL_3RD_PARTY_INCLUDE_DIRS})
set(CGAL_LIBRARIES ${CGAL_LIBRARY} ${CGAL_3RD_PARTY_LIBRARIES})

# Add GMP and MPFR libraries if defined by CGAL
if (GMP_LIBRARIES)
  set(CGAL_LIBRARIES ${CGAL_LIBRARIES} ${GMP_LIBRARIES})
endif()
if (MPFR_LIBRARIES)
  set(CGAL_LIBRARIES ${CGAL_LIBRARIES} ${MPFR_LIBRARIES})
endif()

# Try compiling and running test program
if (DOLFIN_SKIP_BUILD_TESTS)
  set(CGAL_TEST_RUNS TRUE)
elseif (CGAL_INCLUDE_DIRS AND CGAL_LIBRARIES)

  # Set flags for building test program
  set(CMAKE_REQUIRED_INCLUDES ${CGAL_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${CGAL_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${CGAL_CXX_FLAGS_INIT})

  # Add all previusly found Boost libraries - CGAL doesn't appear to supply
  # all necessary Boost libs (test with Boost 1.50 + CGAL 4.0.2)
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${Boost_LIBRARIES})

  # Build and run test program
  include(CheckCXXSourceRuns)
  check_cxx_source_runs("
// CGAL test program
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Point_3.h>
#include <CGAL/Polyhedron_3.h>
typedef CGAL::Simple_cartesian<double> SCK;
typedef SCK::Point_3 Point;
typedef CGAL::Polyhedron_3<SCK> Polyhedron_3;
int main()
{
  // CGAL points
  Point p1(0, 0, 0);
  Point p2(1, 0, 0);
  Point p3(0, 1, 0);
  Point p4(0, 0, 1);
  Polyhedron_3 P;
  P.make_tetrahedron(p1, p2, p3, p4);
  return 0;
}
" CGAL_TEST_RUNS)

endif()

# Standard package handling
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGAL
  "CGAL could not be found. Be sure to set CGAL_DIR"
  CGAL_LIBRARIES CGAL_INCLUDE_DIRS CGAL_TEST_RUNS CGAL_VERSION_OK)