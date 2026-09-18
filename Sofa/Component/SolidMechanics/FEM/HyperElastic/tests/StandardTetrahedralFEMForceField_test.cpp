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
#include <sofa/component/solidmechanics/fem/hyperelastic/StandardTetrahedralFEMForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::fem::hyperelastic
{

using StandardTetrahedralFEMFF = sofa::component::solidmechanics::fem::hyperelastic::StandardTetrahedralFEMForceField<defaulttype::Vec3Types>;

class StandardTetrahedralFEMForceFieldTest : public StandardTetrahedralFEMFF
{
public:
    using TetrahedronRestInformation = StandardTetrahedralFEMFF::TetrahedronRestInformation;
    using EdgeInformation = StandardTetrahedralFEMFF::EdgeInformation;
};

TEST(StandardTetrahedralFEMForceFieldTest, TetrahedronRestInformationStreamOperators)
{

    StandardTetrahedralFEMForceFieldTest::TetrahedronRestInformation initialInfo;

    initialInfo.restVolume = 1;
    initialInfo.volScale = 1;

    for (int i = 0; i < 4; ++i)
        initialInfo.shapeVector[i] = StandardTetrahedralFEMForceFieldTest::Coord();

    initialInfo.fiberDirection = StandardTetrahedralFEMForceFieldTest::Coord();

    for (int i = 0; i < 4; ++i)
        initialInfo.dJ[i] = StandardTetrahedralFEMForceFieldTest::Coord();

    initialInfo.strainEnergy = 1;

    for (int i = 0; i < 4; ++i)
        initialInfo.tetraIndices[i] = 1;

    for (int i = 0; i < 6; ++i)
        initialInfo.tetraEdges[i] = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    StandardTetrahedralFEMForceFieldTest::TetrahedronRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.restVolume, loadedInfo.restVolume);
    EXPECT_EQ(initialInfo.volScale, loadedInfo.volScale);
    EXPECT_EQ(*(initialInfo.shapeVector), *(loadedInfo.shapeVector));
    EXPECT_EQ(initialInfo.fiberDirection, loadedInfo.fiberDirection);
    EXPECT_EQ(*(initialInfo.dJ), *(loadedInfo.dJ));
    EXPECT_EQ(initialInfo.strainEnergy, loadedInfo.strainEnergy);
    EXPECT_EQ(*(initialInfo.tetraIndices), *(loadedInfo.tetraIndices));
    EXPECT_EQ(*(initialInfo.tetraEdges), *(loadedInfo.tetraEdges));

}



}
