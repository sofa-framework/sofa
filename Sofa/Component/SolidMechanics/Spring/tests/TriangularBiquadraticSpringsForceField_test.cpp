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
#include <sofa/component/solidmechanics/spring/TriangularBiquadraticSpringsForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::spring
{

using TriangularBiquadraticBS = sofa::component::solidmechanics::spring::TriangularBiquadraticSpringsForceField<defaulttype::Vec3Types>;

class TriangularBiquadraticBendingSpringsTest : public TriangularBiquadraticBS
{
public:
    using EdgeRestInformation = TriangularBiquadraticBS::EdgeRestInformation;
    using TriangleRestInformation = TriangularBiquadraticBS::TriangleRestInformation;
};

TEST(TriangularBiquadraticBendingSpringsTest, EdgeRestInformationStreamOperators)
{

    TriangularBiquadraticBendingSpringsTest::EdgeRestInformation initialInfo;

    initialInfo.restSquareLength = 1;
    initialInfo.currentSquareLength = 1;
    initialInfo.deltaL2 = 1;
    initialInfo.stiffness = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularBiquadraticBendingSpringsTest::EdgeRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.restSquareLength, loadedInfo.restSquareLength);
    EXPECT_EQ(initialInfo.currentSquareLength, loadedInfo.currentSquareLength);
    EXPECT_EQ(initialInfo.deltaL2, loadedInfo.deltaL2);
    EXPECT_EQ(initialInfo.stiffness, loadedInfo.stiffness);

}

TEST(TriangularBiquadraticBendingSpringsTest, TriangleRestInformationStreamOperators)
{

    TriangularBiquadraticBendingSpringsTest::TriangleRestInformation initialInfo;

    for (int i = 0; i < 3; ++i)
        initialInfo.gamma[i] = i;
    for (int i = 0; i < 3; ++i)
        initialInfo.stiffness[i] = i;
    for (int i = 0; i < 3; ++i)
        initialInfo.DfDx[i] = TriangularBiquadraticBendingSpringsTest::Mat3();

    initialInfo.currentNormal = TriangularBiquadraticBendingSpringsTest::Coord();
    initialInfo.lastValidNormal = TriangularBiquadraticBendingSpringsTest::Coord();
    initialInfo.area = 1;
    initialInfo.restArea = 1;

    for (int i = 0; i < 3; ++i)
        initialInfo.areaVector[i] = TriangularBiquadraticBendingSpringsTest::Coord();
    for (int i = 0; i < 3; ++i)
        initialInfo.dp[i] = TriangularBiquadraticBendingSpringsTest::Deriv();
    initialInfo.J = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularBiquadraticBendingSpringsTest::TriangleRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(*(initialInfo.gamma), *(loadedInfo.gamma));
    EXPECT_EQ(*(initialInfo.stiffness), *(loadedInfo.stiffness));
    EXPECT_EQ(*(initialInfo.DfDx), *(loadedInfo.DfDx));
    EXPECT_EQ(initialInfo.currentNormal, loadedInfo.currentNormal);
    EXPECT_EQ(initialInfo.lastValidNormal, loadedInfo.lastValidNormal);
    EXPECT_EQ(initialInfo.area, loadedInfo.area);
    EXPECT_EQ(initialInfo.restArea, loadedInfo.restArea);
    EXPECT_EQ(*(initialInfo.areaVector), *(loadedInfo.areaVector));
    EXPECT_EQ(*(initialInfo.dp), *(loadedInfo.dp));
    EXPECT_EQ(initialInfo.J, loadedInfo.J);
}

}
