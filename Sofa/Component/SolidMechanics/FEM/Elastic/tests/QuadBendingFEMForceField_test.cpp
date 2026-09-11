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
#include <sofa/component/solidmechanics/fem/elastic/QuadBendingFEMForceField.h>

#include <gtest/gtest.h>
#include <sstream>

namespace sofa
{

using QuadBendingFEMFF = sofa::component::solidmechanics::fem::elastic::QuadBendingFEMForceField<defaulttype::Vec3Types>;

class QuadBendingFEMForceFieldTest : public QuadBendingFEMFF
{
public:
    using QuadInformation = QuadBendingFEMFF::QuadInformation;
    using EdgeInformation = QuadBendingFEMFF::EdgeInformation;
    using VertexInformation = QuadBendingFEMFF::VertexInformation;
};

TEST(QuadBendingFEMForceFieldTest, QuadInformationStreamOperators)
{
    QuadBendingFEMForceFieldTest::QuadInformation initialInfo;

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            initialInfo.BendingmaterialMatrix[i][j] = i + j;

    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
            initialInfo.ShearmaterialMatrix[i][j] = i + j;

    for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 20; ++j)
            initialInfo.strainDisplacementMatrix[i][j] = i + j;

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            initialInfo.stiffness[i][j] = i + j;

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            initialInfo.Bendingstiffness[i][j] = i + j;

    for (int i = 0; i < 20; ++i)
        for (int j = 0; j < 20; ++j)
            initialInfo.Shearstiffness[i][j] = i + j;

    for (int i = 0; i < 3; ++i)
        initialInfo.InitialPosElements[i] = QuadBendingFEMFF::Coord(i, i + 1, i + 2);

    initialInfo.IntlengthElement = QuadBendingFEMFF::Coord(1, 2, 3);

    initialInfo.IntheightElement = QuadBendingFEMFF::Coord(4, 5, 6);

    initialInfo.Intcentroid = QuadBendingFEMFF::Coord(7, 8, 9);

    initialInfo.Inthalflength = 10;
    initialInfo.Inthalfheight = 20;
    initialInfo.differenceToCriteria = 30;

    std::stringstream buffer;
    buffer << initialInfo;

    QuadBendingFEMForceFieldTest::QuadInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(loadedInfo.BendingmaterialMatrix,initialInfo.BendingmaterialMatrix);
    EXPECT_EQ(loadedInfo.ShearmaterialMatrix,initialInfo.ShearmaterialMatrix);
    EXPECT_EQ(loadedInfo.strainDisplacementMatrix,initialInfo.strainDisplacementMatrix);
    EXPECT_EQ(loadedInfo.stiffness,initialInfo.stiffness);
    EXPECT_EQ(loadedInfo.Bendingstiffness,initialInfo.Bendingstiffness);
    EXPECT_EQ(loadedInfo.Shearstiffness,initialInfo.Shearstiffness);
    EXPECT_EQ(loadedInfo.InitialPosElements,initialInfo.InitialPosElements);
    EXPECT_EQ(loadedInfo.IntlengthElement,initialInfo.IntlengthElement);
    EXPECT_EQ(loadedInfo.IntheightElement,initialInfo.IntheightElement);
    EXPECT_EQ(loadedInfo.Intcentroid,initialInfo.Intcentroid);
    EXPECT_EQ(loadedInfo.Inthalflength,initialInfo.Inthalflength);
    EXPECT_EQ(loadedInfo.Inthalfheight,initialInfo.Inthalfheight);
    EXPECT_EQ(loadedInfo.differenceToCriteria,initialInfo.differenceToCriteria);
}

TEST(QuadBendingFEMForceFieldTest, EdgeInformationStreamOperators)
{
    QuadBendingFEMForceFieldTest::EdgeInformation initialInfo;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.DfDx[i][j] = i + j;

    initialInfo.m1 = 10;
    initialInfo.m2 = 20;

    initialInfo.ks = 100;
    initialInfo.kd = 200;
    initialInfo.restlength = 300;

    initialInfo.is_activated = true;
    initialInfo.is_initialized = true;

    std::stringstream buffer;
    buffer << initialInfo;

    QuadBendingFEMForceFieldTest::EdgeInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(loadedInfo.DfDx,initialInfo.DfDx);
    EXPECT_EQ(loadedInfo.m1,initialInfo.m1);
    EXPECT_EQ(loadedInfo.m2,initialInfo.m2);
    EXPECT_EQ(loadedInfo.ks,initialInfo.ks);
    EXPECT_EQ(loadedInfo.kd,initialInfo.kd);
    EXPECT_EQ(loadedInfo.restlength,initialInfo.restlength);
    EXPECT_EQ(loadedInfo.is_activated,initialInfo.is_activated);
    EXPECT_EQ(loadedInfo.is_initialized,initialInfo.is_initialized);
}

TEST(QuadBendingFEMForceFieldTest, VertexInformationStreamOperators)
{
    QuadBendingFEMForceFieldTest::VertexInformation initialInfo;

    initialInfo.meanStrainDirection = QuadBendingFEMFF::Coord(1,2,3);
    initialInfo.sumEigenValues = 1;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.rotation[i][j] = i + j;

    initialInfo.stress = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    QuadBendingFEMForceFieldTest::VertexInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(loadedInfo.meanStrainDirection,initialInfo.meanStrainDirection);
    EXPECT_EQ(loadedInfo.sumEigenValues,initialInfo.sumEigenValues);
    EXPECT_EQ(loadedInfo.rotation,initialInfo.rotation);
    EXPECT_EQ(loadedInfo.stress,initialInfo.stress);
}
} // namespace sofa
