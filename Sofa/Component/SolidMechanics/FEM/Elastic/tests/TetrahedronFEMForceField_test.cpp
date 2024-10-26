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
#include <sofa/component/solidmechanics/fem/elastic/TetrahedronFEMForceField.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

#include "BaseTetrahedronFEMForceField_test.h"

namespace sofa
{

using TetrahedronFEMForceField3 = sofa::component::solidmechanics::fem::elastic::TetrahedronFEMForceField<sofa::defaulttype::Vec3Types>;

INSTANTIATE_TYPED_TEST_SUITE_P(
    TetrahedronFEMForceField_test,
    BaseTetrahedronFEMForceField_test,
    TetrahedronFEMForceField3
);

class TetrahedronFEMForceField_test : public BaseTetrahedronFEMForceField_test<TetrahedronFEMForceField3>
{
public:
    void computeMatricesCheckInit(Transformation& initRot, Transformation& curRot, MaterialStiffness& stiffnessMat, StrainDisplacement& strainD, TetraCoord& initPosition, sofa::Size elementId) override
    {
        typename TetrahedronFEMForceField3::SPtr tetraFEM = m_root->getTreeObject<TetrahedronFEMForceField3>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        initRot = tetraFEM->getInitialTetraRotation(elementId);
        initPosition = tetraFEM->getRotatedInitialElements(elementId);

        curRot = tetraFEM->getActualTetraRotation(elementId);

        stiffnessMat = tetraFEM->getMaterialStiffness(elementId);
        strainD = tetraFEM->getStrainDisplacement(elementId);
    }

    void checkGracefullHandlingWhenTopologyIsMissing()
    {
        // This is a RAII message.
        EXPECT_MSG_EMIT(Error) ;

        std::stringstream scene ;
        scene << R"scn(
<?xml version='1.0'?>
<Node name="root">
    <RequiredPlugin name="Sofa.Component.StateContainer"/>
    <RequiredPlugin name="Sofa.Component.SolidMechanics.FEM.Elastic"/>
    <DefaultAnimationLoop/>
    <Node name="FEMnode">
        <MechanicalObject/>
        <TetrahedronFEMForceField name="fem" youngModulus="5000" poissonRatio="0.07" printLog="true"/>
    </Node>
</Node>
)scn";

        m_root = simulation::SceneLoaderXML::loadFromMemory ("testscene", scene.str().c_str()) ;
        m_root->init(sofa::core::execparams::defaultInstance());

        core::objectmodel::BaseObject* fem = m_root->getTreeNode("FEMnode")->getObject("fem") ;
        EXPECT_NE(fem, nullptr) ;

        EXPECT_EQ(fem->getComponentState(), core::objectmodel::ComponentState::Invalid) ;
    }
};

TEST_F(TetrahedronFEMForceField_test, init)
{
    this->checkInit();
}

TEST_F(TetrahedronFEMForceField_test, FEMValues)
{
    this->checkFEMValues();
}

TEST_F(TetrahedronFEMForceField_test, emptyTology)
{
    this->checkEmptyTopology();
}

TEST_F(TetrahedronFEMForceField_test, checkGracefullHandlingWhenTopologyIsMissing)
{
    this->checkGracefullHandlingWhenTopologyIsMissing();
}

} // namespace sofa
