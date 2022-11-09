/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <sofa/helper/DiffLib.h>

namespace sofa
{

class DiffLib_test : public BaseTest
{
public:
    void getClosestMatch()
    {
        const auto& res = sofa::helper::getClosestMatch("MecahnicaObject", {"MechanicalObject", "Potatoes",
                                                                            "MechanicalState", "Sprout"});
        ASSERT_EQ( res.size(),  2 );
        ASSERT_EQ( std::get<0>(res[0]), "MechanicalObject" );
        ASSERT_EQ( std::get<0>(res[1]), "MechanicalState" );
    }
};

TEST_F(DiffLib_test, normalBehavior_getClosestMatch)
{
    getClosestMatch();
}

} //namespace sofa
