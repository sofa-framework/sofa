/** \mainpage Embryo of a SOFA test suite
  This test suite uses the Boost Unit Testing Framework. http://www.boost.org/doc/libs/1_49_0/libs/test/doc/html/index.html

  The main() function is in an external library. Installatations instructions can be found on the web, e.g.:
  - for linux: http://www.alittlemadness.com/2009/03/31/c-unit-testing-with-boosttest/
  - for windows: http://www.beroux.com/english/articles/boost_unit_testing/

  A good introduction can be found in: http://www.ibm.com/developerworks/aix/library/au-ctools1_boost/

  The test suite includes:
  - sparse matrix tests in MatrixTest.cpp

  */

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>


