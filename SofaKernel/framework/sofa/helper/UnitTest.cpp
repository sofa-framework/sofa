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
#include "UnitTest.h"
#include <time.h>
#include <iostream>

namespace sofa
{

namespace helper
{


UnitTest::UnitTest( std::string testName, VerbosityLevel verb )
{
    name = testName;
    verbose = verb;
}

bool UnitTest::checkIf( bool testSucceeded, std::string testDescription, unsigned& ntests, unsigned& nerr)
{
    ntests++;
    if( !testSucceeded ) nerr++;
    if( testSucceeded )
    {
        sout()  << "---- SUCCESS of : " << testDescription << std::endl;
    }
    else
    {
        serr() <<  "==== FAILURE of : " << testDescription << std::endl;
    }
    return testSucceeded;
}

void UnitTest::initClass()
{
    srand( (unsigned int)time(NULL) ); // initialize the random number generator using the current time
}




} // namespace helper

} // namespace sofa

