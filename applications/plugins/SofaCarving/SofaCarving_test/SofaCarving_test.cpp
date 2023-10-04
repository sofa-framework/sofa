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
#include <sofa/helper/system/FileRepository.h>
#include <SofaCarving/CarvingManager.h>
#include <sofa/simulation/graph/SimpleApi.h>
#include <sofa/core/topology/BaseMeshTopology.h>
#include <sofa/testing/BaseSimulationTest.h>

using namespace sofa::testing;
using namespace sofa::component::collision;
using namespace sofa::simpleapi;

/// <summary>
/// Class to test main methods and behavior of CarvingManager.
/// </summary>
class SofaCarving_test : public BaseSimulationTest
{
public:
    SofaCarving_test()
        : BaseSimulationTest()
    {
        sofa::helper::system::DataRepository.addFirstPath(SOFACARVING_TEST_RESOURCES_DIR);
    }

    
    /// Test creation of CarvingManager outside from a simulation scene
    void ManagerEmpty();
    /// Test creation and empty init of CarvingManager in a basic scene and check componentState
    void ManagerInit();
    /// Test creation and init with links of CarvingManager in a basic scene and check componentState
    void ManagerInitWithLinks();
    
    /// Test wrong init of CarvingManager in a basic scene and check componentState
    void ManagerWrongInit();

    
    /// Test creation and init of CarvingManager in full scene. Will use @sa createScene
    void ManagerSceneInit();
    /// Test carving process with default parameters. Will check topology after carving.
    void doCarving();
    /// Test carving process with penetration parameters. Will check topology after carving.
    void doCarvingWithPenetration();

    /// Unload the scene
    void TearDown() override
    {
        if (m_simu != nullptr && m_root != nullptr) {
            sofa::simulation::node::unload(m_root);
        }
    }

protected:
    /// Method to create full carving scene on a deformable object.
    bool createScene(const std::string& carvingDistance);

private:
    /// Pointer to SOFA simulation
    sofa::simulation::Simulation::SPtr m_simu = nullptr;
    /// Pointer to root Node
    sofa::simulation::Node::SPtr m_root = nullptr;
};


bool SofaCarving_test::createScene(const std::string& carvingDistance)
{
    sofa::simpleapi::importPlugin("Sofa.Component");

    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");
   
    // set scene variables
    m_root->setGravity(sofa::type::Vec3(0.0, 0.0, -0.9));
    m_root->setDt(0.01);

    // create collision pipeline
    createObject(m_root, "DefaultAnimationLoop", { { "name","DefaultAnimationLoop " } });
    createObject(m_root, "CollisionPipeline", { { "name","Collision Pipeline" } });
    createObject(m_root, "BruteForceBroadPhase", { { "name","Broad Phase Detection" } });
    createObject(m_root, "BVHNarrowPhase", { { "name","Narrow Phase Detection" } });
    createObject(m_root, "CollisionResponse", {
        { "name", "Contact Manager" },
        { "response", "PenalityContactForceField" }
    });
    createObject(m_root, "LocalMinDistance", { { "name","localmindistance" },
        { "alarmDistance", "0.5" },
        { "contactDistance", "0.1" }
    });
    

    // create solver
    createObject(m_root, "EulerImplicitSolver", { { "name","Euler Implicit" },
        { "rayleighStiffness","0.1" },
        { "rayleighMass", "0.1" }
    });
    createObject(m_root, "CGLinearSolver", { { "name","Conjugate Gradient" },
        { "iterations","25" },
        { "threshold", "1e-9" },
        { "tolerance", "1e-9" } 
    });
    
    // create carving
    createObject(m_root, "CarvingManager", { { "name","Carving Manager" },
        { "active","1" },
        { "carvingDistance", carvingDistance }
        }
    );

    // create cylinder object
    Node::SPtr nodeVolume = createChild(m_root, "cylinder");
    createObject(nodeVolume, "MeshGmshLoader", {
        { "name","loader" },
        { "filename", sofa::helper::system::DataRepository.getFile("mesh/cylinder.msh") }
    });

    createObject(nodeVolume, "MechanicalObject", {
        { "name","Volume" },
        { "template","Vec3" },
        { "src", "@loader" }
    });


    createObject(nodeVolume, "TetrahedronSetTopologyContainer", {
        { "name","Container" },
        { "src", "@loader" }
        });
    createObject(nodeVolume, "TetrahedronSetTopologyModifier", {
        { "name","Modifier" }
        });
    createObject(nodeVolume, "TetrahedronSetGeometryAlgorithms", {
        { "name","GeomAlgo" },
        { "template", "Vec3" }
        });

    
    createObject(nodeVolume, "DiagonalMass", {
        { "name", "mass" },
        { "massDensity", "0.01" } 
    });
    
    createObject(nodeVolume, "BoxROI", {
        { "name", "ROI1" },
        { "template", "Vec3" },
        { "box", "-1 -1 -1 1 1 0.01" }
    });
    createObject(nodeVolume, "FixedConstraint", {
        { "name", "FIX1" },
        { "indices", "@ROI1.indices" }
    });
    createObject(nodeVolume, "TetrahedralCorotationalFEMForceField", {
        { "name", "CFEM" },
        { "poissonRatio", "0.3" },
        { "method", "large" },
        { "youngModulus", "300" }
    });


    // create cylinder surface
    Node::SPtr nodeSurface = createChild(nodeVolume, "Surface");

    createObject(nodeSurface, "TriangleSetTopologyContainer", {
        { "name","Container" }
        });
    createObject(nodeSurface, "TriangleSetTopologyModifier", {
        { "name","Modifier" }
        });
    createObject(nodeSurface, "TriangleSetGeometryAlgorithms", {
        { "name","GeomAlgo" },
        { "template", "Vec3" }
        });

    createObject(nodeSurface, "Tetra2TriangleTopologicalMapping", {
        { "name","topoMap" },
        { "input", "@../Container" },
        { "output", "@Container" }
        });

    createObject(nodeSurface, "TriangleCollisionModel", {
        { "name", "Triangle Model" },
        { "tags", "CarvingSurface" }
        });

    createObject(nodeSurface, "PointCollisionModel", {
        { "name", "Point Model" },
        { "tags", "CarvingSurface" }
        });



    // create carving Node
    Node::SPtr nodeCarv = createChild(m_root, "carvingElement");

    createObject(nodeCarv, "MechanicalObject", {
        { "name","Particles" },
        { "template","Vec3" },
        { "position", "0 0 1.0" },
        { "velocity", "0 0 0" }
    });

    createObject(nodeCarv, "UniformMass", {
        { "name","Mass" },
        { "totalMass", "1.0" }
    });

    createObject(nodeCarv, "SphereCollisionModel", {
        { "name", "Sphere Model" },
        { "radius", "0.02" },
        { "tags", "CarvingTool" }
        });
        
    return true;
}


void SofaCarving_test::ManagerEmpty()
{
    CarvingManager::SPtr carvingMgr = sofa::core::objectmodel::New< CarvingManager >();
    carvingMgr->doCarve(); // expect nothing to be done nor crash.

    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
}


void SofaCarving_test::ManagerInit()
{
    sofa::simpleapi::importPlugin("Sofa.Component");

    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");

    // create collision pipeline
    createObject(m_root, "DefaultAnimationLoop", { { "name","DefaultAnimationLoop " } });
    createObject(m_root, "CollisionPipeline", { { "name","Collision Pipeline" } });
    createObject(m_root, "CollisionResponse", { { "response","PenalityContactForceField" } });
    createObject(m_root, "BruteForceBroadPhase", { { "name","broadPhase" } });
    createObject(m_root, "BVHNarrowPhase", { { "name","narrowPhase" } });
    createObject(m_root, "MinProximityIntersection", { { "name","Proximity" },
        { "alarmDistance", "0.5" },
        { "contactDistance", "0.02" }
    });

    // create carving
    createObject(m_root, "CarvingManager", { { "name","Carving Manager" },
        { "active","1" },
        { "carvingDistance", "0.1" }
    });


    // create empty collision model as CarvintTool just for init
    createObject(m_root, "MechanicalObject", {
        { "name","Particles" },
        { "template","Vec3" },
        { "position", "0 0 1.0" },
        { "velocity", "0 0 0" }
    });

    createObject(m_root, "SphereCollisionModel", {
        { "name", "tool" },
        { "template","Vec3" },
        { "radius", "0.02" },
        { "tags", "CarvingTool" },
        { "group", "0" }
    });

    createObject(m_root, "SphereCollisionModel", {
        { "name", "tool" },
        { "template","Vec3" },
        { "radius", "0.02" },
        { "tags", "CarvingSurface" },
        { "group", "1" }
    });

    // init scene
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
    sofa::simulation::node::initRoot(m_root.get());
}


void SofaCarving_test::ManagerInitWithLinks()
{
    sofa::simpleapi::importPlugin("Sofa.Component");

    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");

    // create collision pipeline
    createObject(m_root, "DefaultAnimationLoop", { { "name","DefaultAnimationLoop " } });
    createObject(m_root, "CollisionPipeline", { { "name","Collision Pipeline" } });
    createObject(m_root, "CollisionResponse", { { "response","PenalityContactForceField" } });
    createObject(m_root, "BruteForceBroadPhase", { { "name","broadPhase" } });
    createObject(m_root, "BVHNarrowPhase", { { "name","narrowPhase" } });
    createObject(m_root, "MinProximityIntersection", { { "name","Proximity" },
        { "alarmDistance", "0.5" },
        { "contactDistance", "0.02" }
    });

    // create carving
    createObject(m_root, "CarvingManager", { { "name","Carving Manager" },
        { "active","1" },
        { "carvingDistance", "0.1" },
        { "narrowPhaseDetection", "@narrowPhase" },
        { "toolModel", "@tool" }
    });


    // create empty collision model as CarvintTool just for init
    createObject(m_root, "MechanicalObject", {
        { "name","Particles" },
        { "template","Vec3" },
        { "position", "0 0 1.0" },
        { "velocity", "0 0 0" }
    });

    createObject(m_root, "SphereCollisionModel", {
        { "name", "tool" },
        { "template","Vec3" },
        { "radius", "0.02" },
        { "group", "1" }
    });

    createObject(m_root, "SphereCollisionModel", {
        { "name", "tool" },
        { "template","Vec3" },
        { "radius", "0.02" },
        { "tags", "CarvingSurface" },
        { "group", "1" }
    });

    // init scene
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
    sofa::simulation::node::initRoot(m_root.get());
}


void SofaCarving_test::ManagerWrongInit()
{
    sofa::simpleapi::importPlugin("Sofa.Component");

    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");

    // create collision pipeline
    createObject(m_root, "DefaultAnimationLoop", { { "name","DefaultAnimationLoop " } });
    createObject(m_root, "CollisionPipeline", { { "name","Collision Pipeline" } });
    createObject(m_root, "CollisionResponse", { { "response","PenalityContactForceField" } });
    createObject(m_root, "BruteForceBroadPhase", { { "name","broadPhase" } });
    createObject(m_root, "BVHNarrowPhase", { { "name","narrowPhase" } });
    createObject(m_root, "MinProximityIntersection", { { "name","Proximity" },
        { "alarmDistance", "0.5" },
        { "contactDistance", "0.05" }
    });

    // create carving
    createObject(m_root, "CarvingManager", { { "name","Carving Manager" },
        { "active","1" },
        { "carvingDistance", "0.1" }
    });

    // init scene
    EXPECT_MSG_EMIT(Error);
    sofa::simulation::node::initRoot(m_root.get());
}


void SofaCarving_test::ManagerSceneInit()
{
    bool res = createScene("0.0");
    EXPECT_TRUE(res);

    // init scene
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
    sofa::simulation::node::initRoot(m_root.get());
    
    // get node of the mesh
    sofa::simulation::Node* cylinder = m_root->getChild("cylinder");
    EXPECT_NE(cylinder, nullptr);

    // getting topology
    sofa::core::topology::BaseMeshTopology* topo = cylinder->getMeshTopology();
    EXPECT_NE(topo, nullptr);

    // checking topo at start
    EXPECT_EQ(topo->getNbPoints(), 510);
    EXPECT_EQ(topo->getNbEdges(), 3119);
    EXPECT_EQ(topo->getNbTriangles(), 5040);
    EXPECT_EQ(topo->getNbTetrahedra(), 2430);
}


void SofaCarving_test::doCarving()
{
    bool res = createScene("0.0");
    EXPECT_TRUE(res);

    // init scene
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
    sofa::simulation::node::initRoot(m_root.get());

    // get node of the mesh
    sofa::simulation::Node* cylinder = m_root->getChild("cylinder");
    EXPECT_NE(cylinder, nullptr);

    // getting topology
    sofa::core::topology::BaseMeshTopology* topo = cylinder->getMeshTopology();
    EXPECT_NE(topo, nullptr);

    // perform some steps
    for (unsigned int i = 0; i < 100; ++i)
    {
        sofa::simulation::node::animate(m_root.get(), 0.01);
    }

    // checking topo after carving
    EXPECT_LE(topo->getNbPoints(), 480);
    EXPECT_LE(topo->getNbEdges(), 2900);
    EXPECT_LE(topo->getNbTriangles(), 4500);
    EXPECT_LE(topo->getNbTetrahedra(), 2200);
}


void SofaCarving_test::doCarvingWithPenetration()
{
    bool res = createScene("-0.02");
    EXPECT_TRUE(res);

    // init scene
    EXPECT_MSG_NOEMIT(Error);
    EXPECT_MSG_NOEMIT(Warning);
    sofa::simulation::node::initRoot(m_root.get());

    // get node of the mesh
    sofa::simulation::Node* cylinder = m_root->getChild("cylinder");
    EXPECT_NE(cylinder, nullptr);

    // getting topology
    sofa::core::topology::BaseMeshTopology* topo = cylinder->getMeshTopology();
    EXPECT_NE(topo, nullptr);

    // perform some steps
    for (unsigned int i = 0; i < 100; ++i)
    {
        sofa::simulation::node::animate(m_root.get(), 0.01);
    }

    // checking topo after carving
    EXPECT_LT(topo->getNbPoints(), 510);
    EXPECT_LT(topo->getNbEdges(), 3119);
    EXPECT_LT(topo->getNbTriangles(), 5040);
    EXPECT_LT(topo->getNbTetrahedra(), 2430);
}



TEST_F(SofaCarving_test, testManagerEmpty)
{
    ManagerEmpty();
}

TEST_F(SofaCarving_test, testManagerInit)
{
    ManagerInit();
}

TEST_F(SofaCarving_test, testManagerInitWithLinks)
{
    ManagerInitWithLinks();
}


TEST_F(SofaCarving_test, testManagerWrongInit)
{
    ManagerWrongInit();
}

TEST_F(SofaCarving_test, testCarvingSceneInit)
{
    ManagerSceneInit();
}

TEST_F(SofaCarving_test, testdoCarving)
{
    doCarving();
}

TEST_F(SofaCarving_test, testdoCarvingWithPenetration)
{
    doCarvingWithPenetration();
}


