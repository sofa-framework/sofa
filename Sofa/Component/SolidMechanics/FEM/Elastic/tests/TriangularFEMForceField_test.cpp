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
#include <sofa/component/solidmechanics/fem/elastic/TriangularFEMForceField.h>

#include <gtest/gtest.h>
#include <sstream>

namespace sofa
{

using TriangularFEM = sofa::component::solidmechanics::fem::elastic::TriangularFEMForceField<defaulttype::Vec3Types>;

class TriangularFEMForceFieldTest : public TriangularFEM
{
public:
    typedef typename DataTypes::Coord Coord;
    using TriangleInformation = TriangularFEM::TriangleInformation;
    using VertexInformation = TriangularFEM::VertexInformation;
};

TEST(TriangularFEMForceFieldTest, TriangleInformationStreamOperators)
{
    TriangularFEMForceFieldTest::TriangleInformation initialInfo;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            initialInfo.materialMatrix[i][j] = i * 6 + j;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j)
            initialInfo.strainDisplacementMatrix[i][j] = i * 9 + j;

    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            initialInfo.stiffness[i][j] = i * 9 + j;

    initialInfo.area = 1.0;

    for (int i = 0; i < 3; ++i)
        initialInfo.rotatedInitialElements[i] = TriangularFEMForceFieldTest::Coord(i, i + 1, i + 2);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.rotation[i][j] = i * 3 + j;

    for (int i = 0; i < 3; ++i)
        initialInfo.strain[i] = i + 1;

    for (int i = 0; i < 3; ++i)
        initialInfo.stress[i] = i + 4;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.initialTransformation[i][j] = i + j;

    initialInfo.principalStressDirection = TriangularFEMForceFieldTest::Coord(1.0, 2.0, 3.0);

    initialInfo.maxStress = 10.0;

    initialInfo.principalStrainDirection = TriangularFEMForceFieldTest::Coord(4.0, 5.0, 6.0);

    initialInfo.maxStrain = 20.0;

    initialInfo.differenceToCriteria = 30.0;

    initialInfo.lastNStressDirection.resize(3);

    initialInfo.lastNStressDirection[0] = TriangularFEMForceFieldTest::Coord(1.0, 0.0, 0.0);
    initialInfo.lastNStressDirection[1] = TriangularFEMForceFieldTest::Coord(0.0, 1.0, 0.0);
    initialInfo.lastNStressDirection[2] = TriangularFEMForceFieldTest::Coord(0.0, 0.0, 1.0);

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularFEMForceFieldTest::TriangleInformation loadedInfo;
    buffer >> loadedInfo;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(initialInfo.materialMatrix[i][j], loadedInfo.materialMatrix[i][j]);

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 9; ++j)
            EXPECT_EQ(initialInfo.strainDisplacementMatrix[i][j], loadedInfo.strainDisplacementMatrix[i][j]);

    for (int i = 0; i < 9; ++i)
        for (int j = 0; j < 9; ++j)
            EXPECT_EQ(initialInfo.stiffness[i][j], loadedInfo.stiffness[i][j]);

    EXPECT_EQ(initialInfo.area, loadedInfo.area);

    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(initialInfo.rotatedInitialElements[i], loadedInfo.rotatedInitialElements[i]);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(initialInfo.rotation[i][j], loadedInfo.rotation[i][j]);

    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(initialInfo.strain[i], loadedInfo.strain[i]);

    for (int i = 0; i < 3; ++i)
        EXPECT_EQ(initialInfo.stress[i], loadedInfo.stress[i]);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(initialInfo.initialTransformation[i][j], loadedInfo.initialTransformation[i][j]);

    EXPECT_EQ(initialInfo.principalStressDirection, loadedInfo.principalStressDirection);

    EXPECT_EQ(initialInfo.maxStress, loadedInfo.maxStress);

    EXPECT_EQ(initialInfo.principalStrainDirection, loadedInfo.principalStrainDirection);

    EXPECT_EQ(initialInfo.maxStrain, loadedInfo.maxStrain);

    EXPECT_EQ(initialInfo.differenceToCriteria, loadedInfo.differenceToCriteria);

    ASSERT_EQ(initialInfo.lastNStressDirection.size(), loadedInfo.lastNStressDirection.size());

    for (size_t i = 0; i < initialInfo.lastNStressDirection.size(); ++i)
        EXPECT_EQ(initialInfo.lastNStressDirection[i], loadedInfo.lastNStressDirection[i]);
}

TEST(TriangularFEMForceFieldTest, VertexInformationStreamOperators)
{
    TriangularFEMForceFieldTest::VertexInformation initialInfo;

    initialInfo.meanStrainDirection = TriangularFEMForceFieldTest::Coord(1.0, 2.0, 3.0);
    initialInfo.sumEigenValues = 10.0;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.rotation[i][j] = i * 3 + j;

    initialInfo.stress = 20.0;

    std::stringstream buffer;
    buffer << initialInfo;

    TriangularFEMForceFieldTest::VertexInformation loadedInfo;
    buffer >> loadedInfo;

    EXPECT_EQ(initialInfo.meanStrainDirection, loadedInfo.meanStrainDirection);
    EXPECT_EQ(initialInfo.sumEigenValues, loadedInfo.sumEigenValues);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_EQ(initialInfo.rotation[i][j], loadedInfo.rotation[i][j]);

    EXPECT_EQ(initialInfo.stress, loadedInfo.stress);

}

} // namespace sofa
