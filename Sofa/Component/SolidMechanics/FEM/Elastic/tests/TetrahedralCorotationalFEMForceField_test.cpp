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
#include "BaseTetrahedronFEMForceField_test.h"
#include <sofa/component/solidmechanics/fem/elastic/TetrahedralCorotationalFEMForceField.h>

namespace sofa
{

using TetraCorotationalFEM = sofa::component::solidmechanics::fem::elastic::TetrahedralCorotationalFEMForceField<sofa::defaulttype::Vec3Types>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    TetrahedralCorotationalFEMForceField_test,
    BaseTetrahedronFEMForceField_test,
    TetraCorotationalFEM
);


class TetrahedralCorotationalFEMForceField_test : public BaseTetrahedronFEMForceField_test<TetraCorotationalFEM>
{
public:
    void computeMatricesCheckInit(Transformation& initRot, Transformation& curRot, MaterialStiffness& stiffnessMat, StrainDisplacement& strainD, TetraCoord& initPosition, sofa::Size elementId) override
    {
        typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->d_tetrahedronInfo.getValue()[elementId];
        initRot.transpose(tetraInfo.initialTransformation); // TODO check why transposed is stored in this version
        initPosition = tetraInfo.rotatedInitialElements;

        curRot = initRot;

        stiffnessMat = tetraInfo.materialMatrix;
        strainD = tetraInfo.strainDisplacementTransposedMatrix;
    }

    void computeMatricesCheckFEMValues(Transformation& initRot, Transformation& curRot, MaterialStiffness& stiffnessMat, StrainDisplacement& strainD, TetraCoord& initPosition, sofa::Size elementId) override
    {
        typename TetraCorotationalFEM::SPtr tetraFEM = m_root->getTreeObject<TetraCorotationalFEM>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->d_tetrahedronInfo.getValue()[elementId];
        initRot.transpose(tetraInfo.initialTransformation); // TODO check why transposed is stored in this version
        initPosition = tetraInfo.rotatedInitialElements;

        curRot = tetraInfo.rotation;

        stiffnessMat = tetraInfo.materialMatrix;
        strainD = tetraInfo.strainDisplacementTransposedMatrix;
    }

    sofa::helper::logging::Message::Type expectedMessageWhenEmptyTopology() const override { return sofa::helper::logging::Message::Warning; }
};

TEST_F(TetrahedralCorotationalFEMForceField_test, init)
{
    this->checkInit();
}

TEST_F(TetrahedralCorotationalFEMForceField_test, FEMValues)
{
    this->checkFEMValues();
}

TEST_F(TetrahedralCorotationalFEMForceField_test, emptyTology)
{
    this->checkEmptyTopology();
}

TEST_F(TetrahedralCorotationalFEMForceField_test, TetrahedronInformationStreamOperators)
{
    TetraCorotationalFEM::TetrahedronInformation initialInfo;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            initialInfo.materialMatrix[i][j] = 1;

    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 6; ++j)
            initialInfo.strainDisplacementTransposedMatrix[i][j] = 1;

    for (int i = 0; i < 4; ++i)
        initialInfo.rotatedInitialElements[i] = Coord(1, 2, 3);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            initialInfo.elemShapeFun[i][j] = 1;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.rotation[i][j] = 1;

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            initialInfo.initialTransformation[i][j] = 1;

    std::stringstream buffer;
    buffer << initialInfo;

    TetraCorotationalFEM::TetrahedronInformation loadedInfo;
    buffer >> loadedInfo;

    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(initialInfo.materialMatrix[i][j], loadedInfo.materialMatrix[i][j]);

    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 6; ++j)
            EXPECT_EQ(initialInfo.strainDisplacementTransposedMatrix[i][j],
                     loadedInfo.strainDisplacementTransposedMatrix[i][j]);

    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(initialInfo.rotatedInitialElements[i], loadedInfo.rotatedInitialElements[i]);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            EXPECT_EQ(initialInfo.elemShapeFun[i][j], loadedInfo.elemShapeFun[i][j]);

    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_EQ(initialInfo.rotation[i][j], loadedInfo.rotation[i][j]);
            EXPECT_EQ(initialInfo.initialTransformation[i][j], loadedInfo.initialTransformation[i][j]);
        }
}

}
