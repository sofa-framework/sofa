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
#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/component/solidmechanics/fem/elastic/BeamFEMForceField.h>
#include <sofa/component/topology/container/dynamic/EdgeSetTopologyModifier.h>
#include <sofa/core/topology/TopologyData.inl>

#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/Node.h>
#include <sofa/testing/NumericTest.h>
using sofa::simulation::Node;

#include <sofa/testing/BaseTest.h>
using sofa::testing::BaseTest;

#include <string>
using std::string;


namespace sofa
{
using namespace sofa::defaulttype;
using namespace sofa::simpleapi;
using sofa::component::statecontainer::MechanicalObject;

template <class DataTypes>
class BeamFEMForceField_test : public BaseTest
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef MechanicalObject<DataTypes> MState;
    using BeamFEM = sofa::component::solidmechanics::fem::elastic::BeamFEMForceField<DataTypes>;
    using EdgeModifier = sofa::component::topology::container::dynamic::EdgeSetTopologyModifier;
    typedef typename BeamFEM::BeamInfo BeamInfo;
    typedef typename type::vector<BeamInfo> VecBeamInfo;

    static constexpr const char* rigidTypeName = StdRigidTypes<3, Real>::Name();

protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;
    
public:

    void SetUp() override
    {
        sofa::simpleapi::importPlugin("Sofa.Component");
        m_simulation = sofa::simulation::getSimulation();
    }

    void TearDown() override
    {
        if (m_root != nullptr)
            sofa::simulation::node::unload(m_root);
    }

    void createSimpleBeam(Real radius, Real youngModulus, Real poissonRatio)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        m_root->setGravity(type::Vec3(0.0, -1.0, 0.0));
        m_root->setDt(0.01);

        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "DefaultVisualManagerLoop");

        createObject(m_root, "EulerImplicitSolver");
        createObject(m_root, "CGLinearSolver", { { "iterations", "20" }, { "threshold", "1e-8" }, {"tolerance", "1e-5"} });
        createObject(m_root, "MechanicalObject", {{"template", rigidTypeName}, {"position", "0 0 1 0 0 0 1   1 0 1 0 0 0 1   2 0 1 0 0 0 1   3 0 1 0 0 0 1"} });
        createObject(m_root, "EdgeSetTopologyContainer", { {"edges","0 1  1 2  2 3"} });
        createObject(m_root, "EdgeSetTopologyModifier");
        createObject(m_root, "EdgeSetGeometryAlgorithms", { {"template", rigidTypeName} });

        createObject(m_root, "BeamFEMForceField", { {"Name","Beam"}, {"template", rigidTypeName}, {"radius", str(radius)}, {"youngModulus", str(youngModulus)}, {"poissonRatio", str(poissonRatio)} });
        createObject(m_root, "UniformMass", { {"name","mass"}, {"totalMass","1.0"} });
        createObject(m_root, "FixedProjectiveConstraint", { {"name","fix"}, {"indices","0"} });

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }


    void checkCreation()
    {
        createSimpleBeam(static_cast<Real>(0.05), static_cast<Real>(20000000), static_cast<Real>(0.49));

        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        typename BeamFEM::SPtr bFEM = m_root->getTreeObject<BeamFEM>();
        ASSERT_TRUE(bFEM.get() != nullptr);
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_radius.getValue(), static_cast<Real>(0.05));
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_youngModulus.getValue(), static_cast<Real>(20000000));
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_poissonRatio.getValue(), static_cast<Real>(0.49));
    }


    void checkNoMechanicalObject()
    {
        EXPECT_MSG_EMIT(Error);
        
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "BeamFEMForceField", { {"Name","Beam"}, {"template", rigidTypeName}, {"radius", "0.05"} });
        
        sofa::simulation::node::initRoot(m_root.get());
    }


    void checkNoTopology()
    {
        EXPECT_MSG_EMIT(Error);

        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "DefaultAnimationLoop");
        createObject(m_root, "EulerImplicitSolver");
        createObject(m_root, "CGLinearSolver", { { "iterations", "20" }, { "threshold", "1e-8" }, {"tolerance", "1e-5"} });
        createObject(m_root, "MechanicalObject", { {"template", rigidTypeName}, {"position", "0 0 1 0 0 0 1   1 0 1 0 0 0 1   2 0 1 0 0 0 1   3 0 1 0 0 0 1"} });
        createObject(m_root, "BeamFEMForceField", { {"Name","Beam"}, {"template", rigidTypeName} });

        sofa::simulation::node::initRoot(m_root.get());

        sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
    }


    void checkEmptyTopology()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "MechanicalObject", { {"template",rigidTypeName}, {"position", "0 0 1 0 0 0 1   1 0 1 0 0 0 1   2 0 1 0 0 0 1   3 0 1 0 0 0 1"} });
        createObject(m_root, "EdgeSetTopologyContainer");
        createObject(m_root, "BeamFEMForceField", { {"Name","Beam"}, {"template", rigidTypeName} });

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }


    void checkDefaultAttributes()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "MechanicalObject", { {"template",rigidTypeName}, {"position", "0 0 1 0 0 0 1   1 0 1 0 0 0 1   2 0 1 0 0 0 1   3 0 1 0 0 0 1"} });
        createObject(m_root, "EdgeSetTopologyContainer", { {"edges","0 1  1 2  2 3"} });
        createObject(m_root, "BeamFEMForceField", { {"Name","Beam"}, {"template", rigidTypeName} });

        typename BeamFEM::SPtr bFEM = m_root->getTreeObject<BeamFEM>();
        ASSERT_TRUE(bFEM.get() != nullptr);
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_radius.getValue(), static_cast<Real>(0.1));
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_youngModulus.getValue(), static_cast<Real>(5000));
        ASSERT_FLOATINGPOINT_EQ(bFEM->d_poissonRatio.getValue(), static_cast<Real>(0.49));
    }


    void checkInit()
    {
        static constexpr SReal radius = 0.05_sreal;
        static constexpr SReal young = 20000000_sreal;
        static constexpr SReal poisson = 0.49_sreal;
        createSimpleBeam(radius, young, poisson);

        typename BeamFEM::SPtr bFEM = m_root->getTreeObject<BeamFEM>();
        ASSERT_TRUE(bFEM.get() != nullptr);

        const VecBeamInfo& EdgeInfos = bFEM->m_beamsData.getValue();
        ASSERT_EQ(EdgeInfos.size(), 3);

        // check edgeInfo
        const BeamInfo& bI = EdgeInfos[0];
        ASSERT_EQ(bI._E, young);
        ASSERT_EQ(bI._nu, poisson);
        ASSERT_EQ(bI._r, radius);
        ASSERT_EQ(bI._L, 1.0);
        ASSERT_FLOATINGPOINT_EQ(bI._G, young / (2.0_sreal * (1.0_sreal + poisson)));
    }


    void checkFEMValues()
    {
        Real radius = 0.05;
        Real young = 20000000;
        Real poisson = 0.49;
        createSimpleBeam(radius, young, poisson);

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
     
        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), 4);

        // check positions at init
        EXPECT_NEAR(positions[3][0], 3, 1e-4);
        EXPECT_NEAR(positions[3][1], 0, 1e-4);
        EXPECT_NEAR(positions[3][2], 1, 1e-4);

        // access beam info
        typename BeamFEM::SPtr bFEM = m_root->getTreeObject<BeamFEM>();
        const VecBeamInfo& EdgeInfos = bFEM->m_beamsData.getValue();
        const BeamInfo& bI = EdgeInfos[2];

        // simulate
        for (int i = 0; i < 10; i++)
        {
            sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
        }

        // check positions after simulation
        EXPECT_NEAR(positions[3][0], 3, 1e-4);
        EXPECT_NEAR(positions[3][1], -0.004936, 1e-4);
        EXPECT_NEAR(positions[3][2], 1, 1e-4);

        // check edgeInfo
        ASSERT_EQ(bI._E, young);
        ASSERT_EQ(bI._nu, poisson);
        ASSERT_EQ(bI._r, radius);
        ASSERT_EQ(bI._L, 1.0);
    }


    void checkTopologyChanges()
    {
        createSimpleBeam(0.05, 20000000, 0.49);

        const typename EdgeModifier::SPtr edgeModif = m_root->getTreeObject<EdgeModifier>();
        ASSERT_TRUE(edgeModif.get() != nullptr);

        typename BeamFEM::SPtr bFEM = m_root->getTreeObject<BeamFEM>();
        const VecBeamInfo& EdgeInfos = bFEM->m_beamsData.getValue();

        ASSERT_EQ(EdgeInfos.size(), 3);

        const sofa::topology::SetIndex indices = { 0 };
        edgeModif->removeEdges(indices, true);

        sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
        ASSERT_EQ(EdgeInfos.size(), 2);
    }
};


typedef BeamFEMForceField_test<Rigid3Types> BeamFEMForceField_Rig3_test;

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_Creation)
{
    this->checkCreation();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_noMechanicalObject)
{
    this->checkNoMechanicalObject();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_noTopology)
{
    this->checkNoTopology();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_emptyTopology)
{
    this->checkEmptyTopology();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_defaultAttributes)
{
    this->checkDefaultAttributes();
}

// checkWrongAttributes is missing

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_init)
{
    this->checkInit();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_values)
{
    this->checkFEMValues();
}

TEST_F(BeamFEMForceField_Rig3_test, checkForceField_TopologyChanges)
{
    this->checkTopologyChanges();
}

} // namespace sofa
