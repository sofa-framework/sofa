/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
*                              SOFA :: Framework                              *
*                                                                             *
* Authors: The SOFA Team (see Authors.txt)                                    *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "UnitTest.h"

namespace sofa
{

namespace helper
{

bool UnitTest::verbose=false;

UnitTest::UnitTest( std::string testName )
{
    name = testName;
}

/// Runs the test and return true in case of failure. Optionally print begin and end messages, depending on the verbose variable
bool UnitTest::fails()
{
    if(verbose) std::cerr << "BEGIN " << name << std::endl;
    bool s = succeeds();
    if(verbose || !s)
    {
        std::cerr << msg.str();
        if( s )
            std::cerr << "SUCCESS: " << name << std::endl << std::endl;
        else
            std::cerr << "FAIL: " << name << std::endl << std::endl;
    }
    return !s;
}


} // namespace helper

} // namespace sofa

