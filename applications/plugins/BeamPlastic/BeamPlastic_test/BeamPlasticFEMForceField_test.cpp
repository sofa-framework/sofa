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
#include <string>
using std::string;

#include "../forceField/BeamPlasticFEMForceField.h"

#include <sofa/defaulttype/RigidTypes.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <sofa/simulation/SceneLoaderFactory.h>
using sofa::simulation::SceneLoaderFactory;
using sofa::simulation::SceneLoader;

#include <SofaSimulationGraph/DAGSimulation.h>
#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML;

#include <sofa/helper/system/PluginManager.h>
using sofa::helper::system::PluginManager;

#include <sofa/testing/BaseSimulationTest.h>

namespace sofa
{

using sofa::simulation::Simulation;
using sofa::simulation::Node;
using sofa::component::container::MechanicalObject;

using sofa::defaulttype::Rigid3dTypes;

typedef sofa::testing::BaseSimulationTest BaseSimulationTest;

class BeamPlasticFEMForceField_test : public BaseSimulationTest
{
public:

    typedef BaseSimulationTest::SceneInstance SceneInstance;

    void check_BeamPlasticfEMForceField_init()
    {
        importPlugin("BeamPlastic");

        string scene =
            "<Node name='root' dt='1e-2' gravity='0.0 0.0 0.0'>                                                 "
            "   <MechanicalObject template='Rigid3d' name='DOFs' position='0 0 0 0 0 0 1                        "
            "                                                              5e-4 0 0 0 0 0 1                     "
            "                                                              1e-3 0 0 0 0 0 1                     "
            "                                                              1.5e-3 0 0 0 0 0 1                   "
            "                                                              2e-3 0 0 0 0 0 1                     "
            "                                                              2.5e-3 0 0 0 0 0 1                   "
            "                                                              3e-3 0 0 0 0 0 1                     "
            "                                                              3.5e-3 0 0 0 0 0 1' />               "
            "   <MeshTopology name = 'lines' lines = '0 1 1 2 2 3 3 4 4 5 5 6 6 7' /> '                         "
            "   <BeamPlasticFEMForceField name = 'FEM' poissonRatio = '0.3' youngModulus = '2.03e11'            "
            "                             yieldStress = '4.80e8' zSection = '5e-5' ySection = '5e-5'            "
            "                             usePrecomputedStiffness = 'false'                                     "
            "                             useConsistentTangentOperator = 'false'                                "
            "                             isPerfectlyPlastic = 'false' isTimoshenko = 'true' /> '               "
            "</Node>                                                                                            ";

        SceneInstance testScene = SceneInstance("xml", scene);
        Node::SPtr root = testScene.root;
        ASSERT_NE(root.get(), nullptr);
        //ASSERT_EQ(root.get(), nullptr);
    }
};

// NB: si template -> typedef BeamPlasticFEMForceField_test<Rigid3dTypes> BeamPlasticFEMForceField3_test;

TEST_F(BeamPlasticFEMForceField_test, check_BeamPlasticfEMForceField_init) {
    check_BeamPlasticfEMForceField_init();
}

} // namespace sofa::testing