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
using namespace sofa::simpleapi;
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

    int m_nbrStep = 100;
    //std::vector<int> m_indices = { 0, 20, 40, 1200, 1560 };
    std::vector<int> m_indices = { 3};


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

    void loadFEMScene(int FEMType, int nbrGrid, bool both = false)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        m_root->setGravity(type::Vec3(0.0, 10.0, 0.0));
        m_root->setDt(0.01);

        createObject(m_root, "RegularGridTopology", { {"name", "grid"}, 
            {"n", str(type::Vec3(nbrGrid, nbrGrid, 1))}, {"min", "0 0 0"}, {"max", "10 10 0"} });
        
        unsigned int fixP = 0;
        if (nbrGrid > 1)
            fixP = unsigned int(nbrGrid - 1);
        
        if (both)
        {
            addTriangleFEMNode(FEMType, fixP, "TriangleFEM");
            addTriangleFEMNode(FEMType, fixP, "TriangularFEM");
        }
        else if (FEMType == 0)
        {
            addTriangleFEMNode(FEMType, fixP, "TriangleFEM");
        }
        else if (FEMType == 1)
        {
            addTriangleFEMNode(FEMType, fixP, "TriangularFEM");
        }

        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }

    void addTriangleFEMNode(int FEMType, unsigned int fixP, std::string nodeName)
    {
        Node::SPtr FEMNode = sofa::simpleapi::createChild(m_root, nodeName);
        createObject(FEMNode, "EulerImplicitSolver");
        createObject(FEMNode, "CGLinearSolver", {{ "threshold", "1e-9" }});

        createObject(FEMNode, "MechanicalObject", {
            {"name","dof"}, {"template","Vec3d"}, {"position", "@../grid.position"} });

        createObject(FEMNode, "TriangleSetTopologyContainer", {
            {"name","topo"}, {"src","@../grid"} });
        createObject(FEMNode, "TriangleSetTopologyModifier", {
            {"name","Modifier"} });
        createObject(FEMNode, "TriangleSetGeometryAlgorithms", {
            {"name","GeomAlgo"}, {"template","Vec3d"} });

        if (FEMType == 0) // TriangleModel
        {
            createObject(FEMNode, "TriangleFEMForceField", {
                {"name","FEM"}, {"youngModulus","100"}, {"poissonRatio","0.3"}, {"method","large"} });
            std::cout << "TriangleFEMForceField" << std::endl;
        }
        else
        {
            std::cout << "TriangularFEMForceField" << std::endl;
            createObject(FEMNode, "TriangularFEMForceField", {
                {"name","FEM"}, {"youngModulus","100"}, {"poissonRatio","0.3"}, {"method","large"} });
        }


        createObject(FEMNode, "DiagonalMass", {
            {"name","mass"}, {"massDensity","0.1"} });
        createObject(FEMNode, "FixedConstraint", {
            {"name","fix"}, {"indices", str(type::Vec2(0, fixP))} });
    }


    void checkTriangleFEMValues()
    {
        // load Triangular FEM
        int nbrGrid = 40;
        loadFEMScene(0, nbrGrid);

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        
        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), nbrGrid * nbrGrid);
        
        
        for (auto id : m_indices)
        {
            std::cout << "init: " << id << " -> " << positions[id] << std::endl;
        }
        
        std::cout << "animate... " << std::endl;

        for (int i = 0; i < m_nbrStep; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }
        
        for (auto id : m_indices)
        {
            std::cout << "end: " << id << " -> " << positions[id] << std::endl;
        }
    }

    void checkTriangularFEMValues()
    {
        // load Triangular FEM
        int nbrGrid = 40;
        loadFEMScene(1, nbrGrid);

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);

        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), nbrGrid * nbrGrid);
        

        for (auto id : m_indices)
        {
            std::cout << "init: " << id << " -> " << positions[id] << std::endl;
        }

        std::cout << "animate... " << std::endl;

        for (int i = 0; i < m_nbrStep; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }

        for (auto id : m_indices)
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
