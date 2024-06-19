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

        const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[elementId];
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

        const typename TetraCorotationalFEM::TetrahedronInformation& tetraInfo = tetraFEM->tetrahedronInfo.getValue()[elementId];
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

}
