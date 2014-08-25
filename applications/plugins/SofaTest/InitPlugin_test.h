/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Plugins                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef INITPlugin_Test_H
#define INITPlugin_Test_H


#include <sofa/helper/system/config.h>

#ifdef SOFA_BUILD_TEST
#define SOFA_TestPlugin_API SOFA_EXPORT_DYNAMIC_LIBRARY
#else
#define SOFA_TestPlugin_API  SOFA_IMPORT_DYNAMIC_LIBRARY
#endif

/** \mainpage
This is the test suite of Sofa. It contains:
- base classes to ease the development of tests, such as sofa::Sofa_test , sofa::Mapping_test
- tests of Sofa classes, in project SofaTest_test/. This currently contains all the tests of the standard (non-plugin) classes. It is far from complete.

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

This is a the starting page of the plugin documentation, defined in file InitPlugin_test.h
  */

#endif // INITPlugin_Test_H
