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
#include <sofa/component/solidmechanics/fem/elastic/HexahedralFEMForceField.h>

#include <gtest/gtest.h>
#include <sstream>

namespace sofa
{

using HexahedralFEMFF = sofa::component::solidmechanics::fem::elastic::HexahedralFEMForceField<defaulttype::Vec3Types>;

class HexahedralFEMForceFieldTest : public HexahedralFEMFF
{
public:
    using HexahedronInformation = HexahedralFEMFF::HexahedronInformation;
};

TEST(HexahedralFEMForceFieldTest, HexahedronInformationStreamOperators)
{
    HexahedralFEMForceFieldTest::HexahedronInformation initialInfo;

    for (int i=0; i<6; i++)
        for (int j=0; j<6; j++)
            initialInfo.materialMatrix[i][j] = i+j;

    for (int i=0; i<8; i++)
        initialInfo.rotatedInitialElements[i] = HexahedralFEMFF::Coord(i,i,i);

    for (int i=0; i<3; i++)
        for (int j=0; j<3; j++)
            initialInfo.rotation[i][j] = i+j;

    for (int i=0; i<24; i++)
        for (int j=0; j<24; j++)
            initialInfo.stiffness[i][j] = i+j;

    std::stringstream buffer;
    buffer << initialInfo;

    HexahedralFEMForceFieldTest::HexahedronInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(loadedInfo.materialMatrix, initialInfo.materialMatrix);
    EXPECT_EQ(loadedInfo.rotatedInitialElements, initialInfo.rotatedInitialElements);
    EXPECT_EQ(loadedInfo.rotation, initialInfo.rotation);
    EXPECT_EQ(loadedInfo.stiffness, initialInfo.stiffness);
}

} // namespace sofa
