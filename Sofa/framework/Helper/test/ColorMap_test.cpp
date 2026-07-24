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
#include <sofa/helper/ColorMap.h>
#include <gtest/gtest.h>

namespace sofa
{


TEST(ColorMap_test, ConstructorDefault)
{
    helper::ColorMap map;
    EXPECT_EQ(map.getNbColors(), 256);
}

TEST(ColorMap_test, ConstructorSizeAndPreset)
{
    helper::ColorMap map(128, helper::ColorMap::ColorPreset::BLUE_TO_RED);
    EXPECT_EQ(map.getNbColors(), 128);
    // BLUE_TO_RED: first color should be blue
    auto c0 = map.getColor(0);
    EXPECT_FLOAT_EQ(c0.r(), 0.0f);
    EXPECT_FLOAT_EQ(c0.g(), 0.0f);
    EXPECT_FLOAT_EQ(c0.b(), 1.0f);
}

TEST(ColorMap_test, ConstructorStringPreset)
{
    helper::ColorMap map(10, "Red to Blue");
    EXPECT_EQ(map.getNbColors(), 10);
    // RED_TO_BLUE: first color should be red
    auto c0 = map.getColor(0);
    EXPECT_FLOAT_EQ(c0.r(), 1.0f);
    EXPECT_FLOAT_EQ(c0.g(), 0.0f);
    EXPECT_FLOAT_EQ(c0.b(), 0.0f);
}

TEST(ColorMap_test, ConstructorTwoColors)
{
    type::RGBAColor c1(1, 0, 0, 1);
    type::RGBAColor c2(0, 0, 1, 1);
    helper::ColorMap map(c1, c2);
    EXPECT_EQ(map.getNbColors(), 2);
    EXPECT_EQ(map.getColor(0), c1);
    EXPECT_EQ(map.getColor(1), c2);
}

TEST(ColorMap_test, BuildFromColorScheme)
{
    helper::ColorMap map;
    bool success = map.buildFromColorScheme(50, helper::ColorMap::ColorPreset::YELLOW_TO_GREEN);
    EXPECT_TRUE(success);
    EXPECT_EQ(map.getNbColors(), 50);
}

TEST(ColorMap_test, Evaluator)
{
    type::RGBAColor c1(0, 0, 0, 1);
    type::RGBAColor c2(1, 1, 1, 1);
    helper::ColorMap map(c1, c2);

    auto eval = map.getEvaluator<float>(0.0f, 1.0f);

    // Test min
    EXPECT_EQ(eval(0.0f), c1);
    // Test max
    EXPECT_EQ(eval(1.0f), c2);
    // Test middle interpolation
    auto mid = eval(0.5f);
    EXPECT_NEAR(mid.r(), 0.5f, 1e-5);
    EXPECT_NEAR(mid.g(), 0.5f, 1e-5);
    EXPECT_NEAR(mid.b(), 0.5f, 1e-5);
    // Test clamping
    EXPECT_EQ(eval(-1.0f), c1);
    EXPECT_EQ(eval(2.0f), c2);
}

TEST(ColorMap_test, StreamOperators)
{
    helper::ColorMap map1(type::RGBAColor(1, 0, 0, 1), type::RGBAColor(0, 1, 0, 1));
    std::stringstream ss;
    ss << map1;

    helper::ColorMap map2;
    ss >> map2;

    EXPECT_EQ(map2.getNbColors(), 2);
    EXPECT_EQ(map2.getColor(0), map1.getColor(0));
    EXPECT_EQ(map2.getColor(1), map1.getColor(1));
}

TEST(ColorMap_test, StreamOperatorPresetName)
{
    std::stringstream ss;
    ss << "Blue to Red";
    helper::ColorMap map;
    ss >> map;

    EXPECT_EQ(map.getNbColors(), 256);
    // Blue to Red starts with Blue
    EXPECT_EQ(map.getColor(0), type::RGBAColor(0, 0, 1, 1));
}


}
