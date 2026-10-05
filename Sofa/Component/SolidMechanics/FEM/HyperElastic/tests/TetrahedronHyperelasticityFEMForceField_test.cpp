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
#include <sofa/component/solidmechanics/fem/hyperelastic/TetrahedronHyperelasticityFEMForceField.h>

#include <gtest/gtest.h>
#include <sstream>

#include <sofa/defaulttype/VecTypes.h>


namespace sofa::component::solidmechanics::fem::hyperelastic
{

using TetrahedronHyperelasticityFEMFF = sofa::component::solidmechanics::fem::hyperelastic::TetrahedronHyperelasticityFEMForceField<defaulttype::Vec3Types>;

class TetrahedronHyperelasticityFEMForceFieldTest : public TetrahedronHyperelasticityFEMFF
{
public:
    using TetrahedronRestInformation = TetrahedronHyperelasticityFEMFF::TetrahedronRestInformation;
    using EdgeInformation = TetrahedronHyperelasticityFEMFF::EdgeInformation;
};

TEST(TetrahedronHyperelasticityFEMForceFieldTest, TetrahedronRestInformationStreamOperators)
{

    TetrahedronHyperelasticityFEMForceFieldTest::TetrahedronRestInformation initialInfo;

    for (int i = 0; i < 4; ++i)
        initialInfo.m_shapeVector[i] = TetrahedronHyperelasticityFEMForceFieldTest::Coord();

    initialInfo.m_fiberDirection = TetrahedronHyperelasticityFEMForceFieldTest::Coord();
    initialInfo.m_restVolume = 1;
    initialInfo.m_volScale = 1;
    initialInfo.m_volume = 1;

    for (int i = 0; i < 3; ++i)
        initialInfo.m_SPKTensorGeneral[i] = i;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.m_deformationGradient[i][j] = i + j;

    initialInfo.m_strainEnergy = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    TetrahedronHyperelasticityFEMForceFieldTest::TetrahedronRestInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(*(initialInfo.m_shapeVector), *(loadedInfo.m_shapeVector));
    EXPECT_EQ(initialInfo.m_fiberDirection, loadedInfo.m_fiberDirection);
    EXPECT_EQ(initialInfo.m_restVolume, loadedInfo.m_restVolume);
    EXPECT_EQ(initialInfo.m_volScale, loadedInfo.m_volScale);
    EXPECT_EQ(initialInfo.m_volume, loadedInfo.m_volume);
    EXPECT_EQ(initialInfo.m_SPKTensorGeneral, loadedInfo.m_SPKTensorGeneral);
    EXPECT_EQ(initialInfo.m_deformationGradient, loadedInfo.m_deformationGradient);
    EXPECT_EQ(initialInfo.m_strainEnergy, loadedInfo.m_strainEnergy);

}

TEST(TetrahedronHyperelasticityFEMForceFieldTest, EdgeInformationStreamOperators)
{
    TetrahedronHyperelasticityFEMForceFieldTest::EdgeInformation initialInfo;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.DfDx[i][j] = i + j;

    std::stringstream buffer;
    buffer << initialInfo;

    TetrahedronHyperelasticityFEMForceFieldTest::EdgeInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.DfDx, loadedInfo.DfDx);
}

}
