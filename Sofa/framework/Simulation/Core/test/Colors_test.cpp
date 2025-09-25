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

#include <sofa/simulation/Colors.h>
using namespace sofa::simulation;

namespace sofa
{

TEST(Colors_test, testColorQueryByID )
{
    auto color = Colors::getColor(Colors::COLORID::NODE);
    ASSERT_STREQ(color, "#dedede");
}

TEST(Colors_test, testColorQueryByIDInvalid )
{
    ASSERT_ANY_THROW(Colors::getColor(Colors::COLORID::ALLCOLORS));
}

TEST(Colors_test, testColorQueryByIDInvalid2 )
{
    ASSERT_ANY_THROW(Colors::getColor(123456));
}

TEST(Colors_test, testColorQueryWithInvalidName )
{
    ASSERT_ANY_THROW(Colors::getColor("ThisIsNotAValidName"));
}

TEST(Colors_test, testColorRegistrationByName )
{
    std::string hex = "#fafafa";
    ASSERT_FALSE( Colors::hasColor("Prefab") );

    Colors::registerColor("Prefab", hex);
    ASSERT_TRUE( Colors::hasColor("Prefab") );

    auto color = Colors::getColor("Prefab");
    ASSERT_STREQ(color, hex.c_str());
}

TEST(Colors_test, testColorOverridesRegistrationByName )
{
    std::string hex = "#fafafa";
    sofa::Index idA = Colors::registerColor("Prefab", hex);
    ASSERT_TRUE(Colors::hasColor("Prefab") );
    ASSERT_STREQ(Colors::getColor("Prefab"), hex.c_str());

    hex = "#fafafb";
    sofa::Index idB= Colors::registerColor("Prefab", hex);
    ASSERT_TRUE(Colors::hasColor("Prefab") );
    ASSERT_STREQ(Colors::getColor("Prefab"), hex.c_str());
    ASSERT_EQ(idA, idB);
}

TEST(Colors_test, testColorRegistrationById )
{
    std::string hex = "#fafafa";
    auto id = Colors::registerColor(hex);
    auto color = Colors::getColor(id);
    ASSERT_STREQ(color, hex.c_str());
}

TEST(Colors_test, testDeprecatedCOLOR )
{
    auto color1 = Colors::getColor(Colors::COLORID::CMODEL);
    auto color2 = Colors::COLOR[Colors::COLORID::CMODEL];
    ASSERT_STREQ(color1, color2);
}

}
