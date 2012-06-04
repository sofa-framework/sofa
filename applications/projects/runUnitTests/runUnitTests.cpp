/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
using sofa::helper::UnitTest;
using std::cerr;
using std::endl;

/// perform unit tests on matrices and return the number of failures
extern int performMatrixTests();

int main(int argc, char** argv)
{
    bool verbose = false;
    sofa::helper::parse("This is a SOFA application. Here are the command line arguments")
    .option(&UnitTest::verbose,'v',"verbose","print execution logs")
    (argc,argv);

    UnitTest::verbose = verbose;
    int numFailedTests = 0;

    if(verbose) cerr << "Begin test suite" << endl;

    numFailedTests += performMatrixTests();

    if( numFailedTests==0 && verbose ) cerr<<"All tests succeeded" << endl;
    if( numFailedTests!=0  )  cerr<< numFailedTests << " tests failed ! " << endl;

    UnitTest::printLogs();

    return numFailedTests;
}



