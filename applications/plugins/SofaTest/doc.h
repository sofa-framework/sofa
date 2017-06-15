/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this program. If not, see <http://www.gnu.org/licenses/>.        *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/

// This file is used to generate the starting page of the doxygen documentation of this SOFA module.
// It should not be included by any external code.
#error doc.h is not meant to be included, it should be read only by doxygen.


/** \mainpage
This is the test suite of Sofa. It contains base classes to ease the development of tests, such as sofa::Sofa_test , sofa::Mapping_test, etc.
The actual tests are implemented in sub-directories of the modules and plugins. For instance, SofaRigid/SofaRigid_test/RigidMapping_test.cpp.

The tests are optional. They are activated using the SOFA-MISC_TESTS flag of the cmake configuration.
This generates a set of executables, each of them running the test suite of the module. For instance, from the debug build directory, run:
 \code{.sh} bin/SofaConstraint_testd \endcode
To run all the tests, run command \code{.sh} ctest --verbose \endcode

Motivation and detail about activation is given in http://wiki.sofa-framework.org/wiki/UnitTesting

The tests are based on the googletest framework http://code.google.com/p/googletest/wiki/Documentation



<h3> Data files</h3>
Some tests require to open data files. These are typically located in the same directory as the test code.
The path to the current directory can be defined in the CMakeLists.txt, and passed by the compiler as a predefined symbol.
For instance, if you set the following line in CMakeLists.txt:

AddCompilerDefinitions("THIS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\\"")

then the following instruction creates a complete, absolute path to fileName:

std::string fileName = std::string(THIS_DIR) + "/" + fileName;

See e.g. SofaTest_test/CMakeLists.txt and SofaTest_test/LoadScene_test.cpp


@author François Faure, Aurélie Dégletagne, and hopefully lots of others !
@date Started in 2013

This is a the starting page of the plugin documentation, defined in file doc.h in the plugin directory.
  */

