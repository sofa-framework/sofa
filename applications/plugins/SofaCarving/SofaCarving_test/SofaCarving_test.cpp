/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaTest/Sofa_test.h>
#include <sofa/helper/system/FileRepository.h>
#include <SofaCarving/CarvingManager.h>
#include <SofaSimulationGraph/SimpleApi.h>
#include <SofaSimulationGraph/testing/BaseSimulationTest.h>
#include <SofaComponentBase/initComponentBase.h>
#include <SofaComponentCommon/initComponentCommon.h>
#include <SofaComponentGeneral/initComponentGeneral.h>
#include <SofaComponentAdvanced/initComponentAdvanced.h>
#include <SofaComponentMisc/initComponentMisc.h>

using namespace sofa::helper::testing;
using namespace sofa::component::collision;
using namespace sofa::simpleapi;
using namespace sofa::simpleapi::components;


class SofaCarving_test : public BaseSimulationTest
{
public:
    SofaCarving_test()
        : BaseSimulationTest()
        , m_simu(NULL)
        , m_root(NULL)
    {
        sofa::helper::system::DataRepository.addFirstPath(SOFACARVING_TEST_RESOURCES_DIR);
    }

    bool createScene(const std::string& carvingDistance);

    bool ManagerEmpty();
    bool ManagerInit();
    bool doCarving();
    bool doCarvingWithPenetration();

private:
    sofa::simulation::Simulation::SPtr m_simu;
    sofa::simulation::Node::SPtr m_root;
};


bool SofaCarving_test::createScene(const std::string& carvingDistance)
{
    sofa::component::initComponentBase();
    sofa::component::initComponentCommon();
    sofa::component::initComponentGeneral();
    sofa::component::initComponentAdvanced();
    sofa::component::initComponentMisc();

    m_simu = createSimulation("DAG");
    m_root = createRootNode(m_simu, "root");
   
    // set scene variables
    m_root->setGravity(sofa::defaulttype::Vector3(0.0, 0.0, -0.9));
    m_root->setDt(0.01);

    // create collision pipeline
    createObject(m_root, "CollisionPipeline", { { "name","Collision Pipeline" } });
    createObject(m_root, "BruteForceDetection", { { "name","Detection" } });
    createObject(m_root, "CollisionResponse", {
        { "name", "Contact Manager" },
        { "response", "default" }
    });
    createObject(m_root, "MinProximityIntersection", { { "name","Proximity" },
        { "alarmDistance", "0.5" },
        { "contactDistance", "0.05" }
    });
    

    // create solver
    createObject(m_root, "EulerImplicitSolver", { { "name","Euler Implicit" },
        { "rayleighStiffness","0.1" },
        { "rayleighMass", "0.1" }
    });
    createObject(m_root, "CGLinearSolver", { { "name","Conjugate Gradient" },
        { "iterations","25" },
        { "threshold", "0.000000001" },
        { "tolerance", "0.000000001" } 
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
    createObject(nodeVolume, "TetrahedronSetTopologyAlgorithms", {
        { "name","TopoAlgo" },
        { "template", "Vec3" }
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
    createObject(nodeSurface, "TriangleSetTopologyAlgorithms", {
        { "name","TopoAlgo" },
        { "template", "Vec3" }
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

    createObject(nodeSurface, "TriangleSet", {
        { "name", "Triangle Model" },
        { "tags", "CarvingSurface" },
        { "group", "0" }
        });

    createObject(nodeSurface, "PointSet", {
        { "name", "Point Model" },
        { "tags", "CarvingSurface" },
        { "group", "0" }
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

    createObject(nodeCarv, "SphereModel", {
        { "name", "Sphere Model" },
        { "radius", "0.02" },
        { "tags", "CarvingTool" },
        { "group", "1" }
        });
        
    return true;
}


bool SofaCarving_test::ManagerEmpty()
{
    CarvingManager::SPtr carvingMgr = sofa::core::objectmodel::New< CarvingManager >();
    carvingMgr->doCarve(); // expect nothing to be done nor crash.

    return true;
}


bool SofaCarving_test::ManagerInit()
{
    bool res = createScene("0.0");
    if (!res)
        return false;

    // init scene
    m_simu->init(m_root.get());
    
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

    return res;
}


bool SofaCarving_test::doCarving()
{
    bool res = createScene("0.0");
    if (!res)
        return false;

    // init scene
    m_simu->init(m_root.get());

    // get node of the mesh
    sofa::simulation::Node* cylinder = m_root->getChild("cylinder");
    EXPECT_NE(cylinder, nullptr);

    // getting topology
    sofa::core::topology::BaseMeshTopology* topo = cylinder->getMeshTopology();
    EXPECT_NE(topo, nullptr);

    // perform some steps
    for (unsigned int i = 0; i < 100; ++i)
    {
        m_simu->animate(m_root.get());
    }

    // checking topo after carving
    EXPECT_LE(topo->getNbPoints(), 480);
    EXPECT_LE(topo->getNbEdges(), 2900);
    EXPECT_LE(topo->getNbTriangles(), 4500);
    EXPECT_LE(topo->getNbTetrahedra(), 2200);
    
    return true;
}


bool SofaCarving_test::doCarvingWithPenetration()
{
    bool res = createScene("-0.02");
    if (!res)
        return false;

    // init scene
    m_simu->init(m_root.get());

    // get node of the mesh
    sofa::simulation::Node* cylinder = m_root->getChild("cylinder");
    EXPECT_NE(cylinder, nullptr);

    // getting topology
    sofa::core::topology::BaseMeshTopology* topo = cylinder->getMeshTopology();
    EXPECT_NE(topo, nullptr);

    // perform some steps
    for (unsigned int i = 0; i < 100; ++i)
    {
        m_simu->animate(m_root.get());
    }

    // checking topo after carving
    EXPECT_LT(topo->getNbPoints(), 510);
    EXPECT_LT(topo->getNbEdges(), 3119);
    EXPECT_LT(topo->getNbTriangles(), 5040);
    EXPECT_LT(topo->getNbTetrahedra(), 2430);

    return true;
}



TEST_F(SofaCarving_test, testManagerEmpty)
{
    ASSERT_TRUE(ManagerEmpty());
}

TEST_F(SofaCarving_test, testManagerInit)
{
    ASSERT_TRUE(ManagerInit());
}

TEST_F(SofaCarving_test, testdoCarving)
{
    ASSERT_TRUE(doCarving());
}

TEST_F(SofaCarving_test, testdoCarvingWithPenetration)
{
    ASSERT_TRUE(doCarvingWithPenetration());
}


