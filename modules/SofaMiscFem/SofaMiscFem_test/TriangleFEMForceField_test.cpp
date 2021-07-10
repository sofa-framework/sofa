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
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaMiscFem/TriangleFEMForceField.h>
#include <SofaMiscFem/TriangularFEMForceField.h>
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>

#include <SofaSimulationGraph/SimpleApi.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <SofaSimulationCommon/SceneLoaderXML.h>
using sofa::simulation::SceneLoaderXML;

#include <string>
using std::string;

namespace sofa
{
using namespace sofa::defaulttype;
using sofa::component::container::MechanicalObject;

template <class DataTypes>
class TriangleFEMForceField_test : public BaseTest
{
public:
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef MechanicalObject<DataTypes> MState;


protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;



public:

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("SofaComponentAll");
        simulation::setSimulation(m_simulation = new simulation::graph::DAGSimulation());
    }

    void TearDown() override
    {
        if (m_root != nullptr)
            simulation::getSimulation()->unload(m_root);
    }


    void loadTriangularFEMScene()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                              "
            "<Node  name='Root' gravity='0 10 0' dt='0.01' animate='0' >                        "
            "    <RegularGridTopology name='grid' n='40 40 1' min='0 0 0' max='10 10 0' />      "
            "    <Node name='FEM' >                                                             "
            "        <EulerImplicitSolver />                                                    "
            "        <CGLinearSolver iterations='25' threshold='1.0e-9'/>                       "
            "        <MechanicalObject name='dof' position='@../grid.position' />               "
            "        <TriangleSetTopologyContainer name='topo' src='@../grid' />                "
            "        <TriangleSetTopologyModifier name='Modifier' />                            "
            "        <TriangleSetGeometryAlgorithms name='GeomAlgo' template='Vec3d' />         "
            "        <TriangularFEMForceField name='FEM' youngModulus='100' poissonRatio='0.3' method='large' /> "
            "        <DiagonalMass massDensity='0.1' />                                        "
            "        <FixedConstraint indices='0 39' />                                        "
            "    </Node>"
            "</Node>  ";

        /// Load scene
        m_root = SceneLoaderXML::loadFromMemory("loadWithNoParam",
            scene.c_str(),
            scene.size());


        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void loadTriangleFEMScene()
    {
        static const string scene =
            "<?xml version='1.0'?>                                                              "
            "<Node  name='Root' gravity='0 10 0' dt='0.01' animate='0' >                        "
            "    <RegularGridTopology name='grid' n='40 40 1' min='0 0 0' max='10 10 0' />      "
            "    <Node name='FEM' >                                                             "
            "        <EulerImplicitSolver />                                                    "
            "        <CGLinearSolver iterations='25' threshold='1.0e-9'/>                       "
            "        <MechanicalObject name='dof' position='@../grid.position' />               "
            "        <TriangleSetTopologyContainer name='topo' src='@../grid' />                "
            "        <TriangleSetTopologyModifier name='Modifier' />                            "
            "        <TriangleSetGeometryAlgorithms name='GeomAlgo' template='Vec3d' />         "
            "        <TriangleFEMForceField name='FEM' youngModulus='100' poissonRatio='0.3' method='large' /> "
            "        <DiagonalMass massDensity='0.1' />                                        "
            "        <FixedConstraint indices='0 39' />                                        "
            "    </Node>"
            "</Node>  ";

        /// Load scene
        m_root = SceneLoaderXML::loadFromMemory("loadWithNoParam",
            scene.c_str(),
            scene.size());


        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void checkTriangleFEMValues()
    {
        std::cout << "-- checkTriangleFEMValues() --" << std::endl;
        // load Triangular FEM
        loadTriangleFEMScene();

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);

        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), 1600);
        std::vector<int> indices = { 0, 20, 40, 1200, 1560 };
        
        for (auto id : indices)
        {
            std::cout << "init: " << id << " -> " << positions[id] << std::endl;
        }
        
        std::cout << "animate... " << std::endl;

        for (int i = 0; i < 100; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }
        
        for (auto id : indices)
        {
            std::cout << "end: " << id << " -> " << positions[id] << std::endl;
        }
    }

    void checkTriangularFEMValues()
    {
        std::cout << "-- checkTriangularFEMValues() --" << std::endl;
        // load Triangular FEM
        loadTriangularFEMScene();

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);

        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), 1600);
        std::vector<int> indices = { 0, 20, 40, 1200, 1560 };

        for (auto id : indices)
        {
            std::cout << "init: " << id << " -> " << positions[id] << std::endl;
        }

        std::cout << "animate... " << std::endl;

        for (int i = 0; i < 100; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }

        for (auto id : indices)
        {
            std::cout << "end: " << id << " -> " << positions[id] << std::endl;
        }
    }


    void compareTriangleFEMValues()
    {

    }

};


typedef TriangleFEMForceField_test<Vec3Types> TriangleFEMForceField3_test;


TEST_F(TriangleFEMForceField3_test, checkTriangleFEMValues)
{
    this->checkTriangleFEMValues();
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMValues)
{
    this->checkTriangularFEMValues();
}


} // namespace sofa
