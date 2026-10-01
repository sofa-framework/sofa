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
#include <sofa/component/solidmechanics/tensormass/TetrahedralTensorMassForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::tensormass
{

using TetrahedralTensorMassFF = sofa::component::solidmechanics::tensormass::TetrahedralTensorMassForceField<defaulttype::Vec3Types>;

class TetrahedralTensorMassForceFieldTest : public TetrahedralTensorMassFF
{
public:
    using EdgeRestInformation = TetrahedralTensorMassFF::EdgeRestInformation;
};

TEST(TetrahedralTensorMassForceFieldTest, EdgeRestInformationStreamOperators)
{
    TetrahedralTensorMassForceFieldTest::EdgeRestInformation initialInfo;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.DfDx[i][j] = i + j;

    for (int i = 0; i < 2; ++i)
        initialInfo.vertices[i] = i;

    std::stringstream buffer;
    buffer << initialInfo;

    TetrahedralTensorMassForceFieldTest::EdgeRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.DfDx, loadedInfo.DfDx);
    EXPECT_EQ(*(initialInfo.vertices), *(loadedInfo.vertices));
}

}
