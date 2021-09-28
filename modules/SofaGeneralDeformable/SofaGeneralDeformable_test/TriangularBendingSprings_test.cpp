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
#include <SofaBaseTopology/TriangleSetTopologyContainer.h>
#include <SofaGeneralDeformable/TriangularBendingSprings.h>
#include <SofaBaseTopology/TopologyData.inl>

#include <SofaSimulationGraph/SimpleApi.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
using sofa::simulation::Node;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <string>
using std::string;


namespace sofa
{
using namespace sofa::defaulttype;
using namespace sofa::simpleapi;
using sofa::component::container::MechanicalObject;

template <class DataTypes>
class TriangularBendingSprings_test : public BaseTest
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef MechanicalObject<DataTypes> MState;
    using TriangleBS = sofa::component::forcefield::TriangularBendingSprings<DataTypes>;
    typedef typename TriangleBS::EdgeInformation EdgeInfo;
    typedef typename type::vector<EdgeInfo> VecEdgeInfo;
    using Vec3 = type::Vec<3, Real>;
    using Mat33 = type::Mat<3, 3, Real>;
    using Mat63 = type::Mat<6, 3, Real>;

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

    void createSimpleTrianglePairScene(Real ks, Real kd)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");        
        createObject(m_root, "MechanicalObject", {{"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0  1 1 1"} });
        createObject(m_root, "TriangleSetTopologyContainer", { {"triangles","0 1 2  1 3 2"} });
        createObject(m_root, "TriangleSetTopologyModifier");
        createObject(m_root, "TriangleSetGeometryAlgorithms", { {"template","Vec3d"} });

        createObject(m_root, "TriangularBendingSprings", { {"Name","TBS"}, {"stiffness", str(ks)}, {"damping", str(kd)} });

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void createGridScene(int nbrGrid, Real ks, Real kd)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        m_root->setGravity(type::Vec3(0.0, -1.0, 0.0));
        m_root->setDt(0.01);

        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");
        createObject(m_root, "RegularGridTopology", { {"name", "grid"}, 
            {"n", str(type::Vec3(nbrGrid, nbrGrid, 1))}, {"min", "0 0 0"}, {"max", "10 10 0"} });
        
        Node::SPtr FNode = sofa::simpleapi::createChild(m_root, "SpringNode");
        createObject(FNode, "EulerImplicitSolver");
        createObject(FNode, "CGLinearSolver", { { "iterations", "20" }, { "threshold", "1e-6" } });
        createObject(FNode, "MechanicalObject", {
            {"name","dof"}, {"template","Vec3d"}, {"position", "@../grid.position"} });
        createObject(FNode, "TriangleSetTopologyContainer", {
            {"name","topo"}, {"src","@../grid"} });
        createObject(FNode, "TriangleSetTopologyModifier", {
            {"name","Modifier"} });
        createObject(FNode, "TriangleSetGeometryAlgorithms", {
            {"name","GeomAlgo"}, {"template","Vec3d"} });
        
        createObject(FNode, "TriangularBendingSprings", { {"Name","TBS"}, {"stiffness", str(ks)}, {"damping", str(kd)} });
        createObject(FNode, "DiagonalMass", { {"name","mass"}, {"massDensity","0.1"} });

        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void checkCreation()
    {
        createSimpleTrianglePairScene(100, 0.1);

        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        typename TriangleBS::SPtr triBS = m_root->getTreeObject<TriangleBS>();
        ASSERT_TRUE(triBS.get() != nullptr);
        ASSERT_FLOAT_EQ(triBS->getKs(), 100);
        ASSERT_FLOAT_EQ(triBS->getKd(), 0.1);
    }


    void checkNoTopology()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "MechanicalObject", { {"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0  1 1 1"} });
        createObject(m_root, "TriangularBendingSprings");

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void checkEmptyTopology()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "MechanicalObject", { {"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0  1 1 1"} });
        createObject(m_root, "TriangleSetTopologyContainer");
        createObject(m_root, "TriangularBendingSprings");

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void checkDefaultAttributes()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "MechanicalObject", { {"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0  1 1 1"} });
        createObject(m_root, "TriangleSetTopologyContainer", { {"triangles","0 1 2  1 3 2"} });
        createObject(m_root, "TriangleSetTopologyModifier");
        createObject(m_root, "TriangleSetGeometryAlgorithms", { {"template","Vec3d"} });
        createObject(m_root, "TriangularBendingSprings");

        typename TriangleBS::SPtr triBS = m_root->getTreeObject<TriangleBS>();
        ASSERT_TRUE(triBS.get() != nullptr);
        ASSERT_FLOAT_EQ(triBS->getKs(), 100000);
        ASSERT_FLOAT_EQ(triBS->getKd(), 1);
    }


    void checkWrongAttributes()
    {
        EXPECT_MSG_EMIT(Warning);
        createSimpleTrianglePairScene(-100, -0.1);
    }


    void checkInit()
    {
        createSimpleTrianglePairScene(300, 0.5);

        typename TriangleBS::SPtr triBS = m_root->getTreeObject<TriangleBS>();
        ASSERT_TRUE(triBS.get() != nullptr);
        
        const VecEdgeInfo& EdgeInfos = triBS->edgeInfo.getValue();
        ASSERT_EQ(EdgeInfos.size(), 5);

        // only one commun edge == only one spring between [0; 3] with restLength = ||(1,1,1)||
        int cptActivated = 0;
        float restLength = sqrtf(3);
        for (auto& ei : EdgeInfos)
        {
            ASSERT_TRUE(ei.is_initialized == true);
            if (ei.is_activated)
            {
                cptActivated++;
                ASSERT_EQ(ei.m1, 3);
                ASSERT_EQ(ei.m2, 0);
                ASSERT_FLOAT_EQ(ei.ks, 300);
                ASSERT_FLOAT_EQ(ei.kd, 0.5);
                ASSERT_FLOAT_EQ(ei.restlength, restLength);
            }
        }

        ASSERT_EQ(cptActivated, 1);
    }


    void checkFEMValues()
    {
        // load Triangular FEM
        int nbrGrid = 20;
        int nbrStep = 10;
        createGridScene(nbrGrid, 300, 0.5);

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        
        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), nbrGrid * nbrGrid);

        typename TriangleBS::SPtr triBS = m_root->getTreeObject<TriangleBS>();
        ASSERT_TRUE(triBS.get() != nullptr);
        ASSERT_FLOAT_EQ(triBS->getAccumulatedPotentialEnergy(), 0.0);

        const VecEdgeInfo& EdgeInfos = triBS->edgeInfo.getValue();
        
        ASSERT_EQ(EdgeInfos.size(), 1121);
        ASSERT_EQ(EdgeInfos[0].is_activated, true);
        ASSERT_EQ(EdgeInfos[1].is_activated, true);
        ASSERT_EQ(EdgeInfos[2].is_activated, false);
        ASSERT_FLOAT_EQ(EdgeInfos[0].restlength, 1.1768779);
        
        EXPECT_NEAR(positions[nbrGrid][0], 0, 1e-4);
        EXPECT_NEAR(positions[nbrGrid][1], 0.5263, 1e-4);
        EXPECT_NEAR(positions[nbrGrid][2], 0, 1e-4);

        for (int i = 0; i < nbrStep; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }

        EXPECT_NEAR(positions[nbrGrid][0], -0.00031, 1e-4);
        EXPECT_NEAR(positions[nbrGrid][1], 0.5216297, 1e-4);
        EXPECT_NEAR(positions[nbrGrid][2], 0, 1e-4);

        ASSERT_FLOAT_EQ(triBS->getAccumulatedPotentialEnergy(), 6.5779859e-05);
        ASSERT_FLOAT_EQ(EdgeInfos[0].restlength, 1.1768779);
    }

};


typedef TriangularBendingSprings_test<Vec3Types> TriangularBendingSprings3_test;

TEST_F(TriangularBendingSprings3_test, checkForceField_Creation)
{
    this->checkCreation();
}

TEST_F(TriangularBendingSprings3_test, checkForceField_noTopology)
{
    this->checkNoTopology();
}

TEST_F(TriangularBendingSprings3_test, checkForceField_emptyTopology)
{
    this->checkEmptyTopology();
}

TEST_F(TriangularBendingSprings3_test, checkForceField_defaultAttributes)
{
    this->checkDefaultAttributes();
}

TEST_F(TriangularBendingSprings3_test, checkFEMForceField_wrongAttributess)
{
    this->checkWrongAttributes();
}

TEST_F(TriangularBendingSprings3_test, checkForceField_init)
{
    this->checkInit();
}

TEST_F(TriangularBendingSprings3_test, checkForceField_values)
{
    this->checkFEMValues();
}

} // namespace sofa
