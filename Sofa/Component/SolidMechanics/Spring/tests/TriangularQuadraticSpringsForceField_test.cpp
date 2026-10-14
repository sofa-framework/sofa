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
#include <sofa/component/solidmechanics/spring/TriangularQuadraticSpringsForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::spring
{

using TriangularQuadraticBS = sofa::component::solidmechanics::spring::TriangularQuadraticSpringsForceField<defaulttype::Vec3Types>;

class TriangularQuadraticBendingSpringsTest : public TriangularQuadraticBS
{
public:
    using EdgeRestInformation = TriangularQuadraticBS::EdgeRestInformation;
    using TriangleRestInformation = TriangularQuadraticBS::TriangleRestInformation;
};

TEST(TriangularQuadraticBendingSpringsTest, EdgeRestInformationStreamOperators)
{

    TriangularQuadraticBendingSpringsTest::EdgeRestInformation initialInfo;

    initialInfo.restLength = 1;
    initialInfo.currentLength = 1;
    initialInfo.dl = 1;
    initialInfo.stiffness = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularQuadraticBendingSpringsTest::EdgeRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.restLength, loadedInfo.restLength);
    EXPECT_EQ(initialInfo.currentLength, loadedInfo.currentLength);
    EXPECT_EQ(initialInfo.dl, loadedInfo.dl);
    EXPECT_EQ(initialInfo.stiffness, loadedInfo.stiffness);

}

TEST(TriangularQuadraticBendingSpringsTest, TriangleRestInformationStreamOperators)
{

    TriangularQuadraticBendingSpringsTest::TriangleRestInformation initialInfo;

    for (int i = 0; i < 3; ++i)
        initialInfo.gamma[i] = i;
    for (int i = 0; i < 3; ++i)
        initialInfo.stiffness[i] = i;
    for (int i = 0; i < 3; ++i)
        initialInfo.DfDx[i] = TriangularQuadraticBendingSpringsTest::Mat3();

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularQuadraticBendingSpringsTest::TriangleRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(*(initialInfo.gamma), *(loadedInfo.gamma));
    EXPECT_EQ(*(initialInfo.stiffness), *(loadedInfo.stiffness));
    EXPECT_EQ(*(initialInfo.DfDx), *(loadedInfo.DfDx));

}

}
