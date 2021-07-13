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

#include <sofa/helper/system/thread/CTime.h>
#include <limits>

namespace sofa
{
using namespace sofa::defaulttype;
using namespace sofa::simpleapi;
using sofa::component::container::MechanicalObject;
using sofa::helper::system::thread::ctime_t;

template <class DataTypes>
class TriangleFEMForceField_test : public BaseTest
{
public:
    typedef typename DataTypes::Real Real;
    typedef typename DataTypes::Coord Coord;
    typedef typename DataTypes::VecCoord VecCoord;
    typedef MechanicalObject<DataTypes> MState;
    using TriangleFEM = sofa::component::forcefield::TriangleFEMForceField<DataTypes>;
    using TriangularFEM = sofa::component::forcefield::TriangularFEMForceField<DataTypes>;


protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;

    int m_nbrStep = 100;
    //std::vector<int> m_indices = { 0, 20, 40, 1200, 1560 };
    std::vector<int> m_indices = { 3};

    ctime_t timeTicks = sofa::helper::system::thread::CTime::getRefTicksPerSec();

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

    void createSingleTriangleFEMScene(int FEMType, float young, float poisson, std::string method)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "MechanicalObject", {{"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0  1 1 1"} });
        createObject(m_root, "TriangleSetTopologyContainer", { {"triangles","0 1 2  1 3 2"} });
        createObject(m_root, "TriangleSetTopologyModifier");
        createObject(m_root, "TriangleSetGeometryAlgorithms", { {"template","Vec3d"} });

        if (FEMType == 0) // TriangleModel
        {
            createObject(m_root, "TriangleFEMForceField", {
                {"name","FEM"}, {"youngModulus", str(young)}, {"poissonRatio", str(poisson)}, {"method", method} });
        }
        else
        {
            createObject(m_root, "TriangularFEMForceField", {
                {"name","FEM"}, {"youngModulus", str(young)}, {"poissonRatio", str(poisson)}, {"method", method} });
        }

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void createGridFEMScene(int FEMType, int nbrGrid, bool both = false)
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
        }
        else
        {
            createObject(FEMNode, "TriangularFEMForceField", {
                {"name","FEM"}, {"youngModulus","100"}, {"poissonRatio","0.3"}, {"method","large"} });
        }


        createObject(FEMNode, "DiagonalMass", {
            {"name","mass"}, {"massDensity","0.1"} });
        createObject(FEMNode, "FixedConstraint", {
            {"name","fix"}, {"indices", str(type::Vec2(0, fixP))} });
    }




    void checkCreation(int FEMType)
    {
        createSingleTriangleFEMScene(FEMType, 100, 0.3, "large");

        MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        if (FEMType == 0)
        {
            TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3);
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 100);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
        else if (FEMType == 1)
        {
            TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3);
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 100);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
    }

    void checkNoTopology(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "MechanicalObject", { {"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0"} });
        if (FEMType == 0) // TriangleModel
        {
            createObject(m_root, "TriangleFEMForceField", {
                {"name","FEM"}, {"youngModulus", "100"}, {"poissonRatio", "0.3"}, {"method", "large"} });
        }
        else
        {
            createObject(m_root, "TriangularFEMForceField", {
                {"name","FEM"}, {"youngModulus", "100"}, {"poissonRatio", "0.3"}, {"method", "large"} });
        }

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }

    void checkEmptyTopology(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        createObject(m_root, "MechanicalObject", { {"template","Vec3d"} });
        createObject(m_root, "TriangleSetTopologyContainer");
        if (FEMType == 0) // TriangleModel
        {
            createObject(m_root, "TriangleFEMForceField", {
                {"name","FEM"}, {"youngModulus", "100"}, {"poissonRatio", "0.3"}, {"method", "large"} });
        }
        else
        {
            createObject(m_root, "TriangularFEMForceField", {
                {"name","FEM"}, {"youngModulus", "100"}, {"poissonRatio", "0.3"}, {"method", "large"} });
        }

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
    }


    void checkDefaultAttributes(int FEMType)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        createObject(m_root, "MechanicalObject", { {"template","Vec3d"}, {"position", "0 0 0  1 0 0  0 1 0"} });
        createObject(m_root, "TriangleSetTopologyContainer", { {"triangles","0 1 2"} });
        createObject(m_root, "TriangleSetTopologyModifier");
        createObject(m_root, "TriangleSetGeometryAlgorithms", { {"template","Vec3d"} });

        if (FEMType == 0) // TriangleModel
        {
            createObject(m_root, "TriangleFEMForceField");
        }
        else if (FEMType == 1)
        {
            createObject(m_root, "TriangularFEMForceField");
        }

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
        if (FEMType == 0)
        {
            TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3);
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 1000);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
        else if (FEMType == 1)
        {
            TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.45); // Not the same default values
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 1000);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
    }


    void checkWrongAttributes(int FEMType)
    {
        EXPECT_MSG_EMIT(Error);
        createSingleTriangleFEMScene(FEMType, -100, -0.3, "toto");
    }


    void checkInit(int FEMType)
    {
        createSingleTriangleFEMScene(FEMType, 100, 0.3, "large");
        type::Mat<3, 3, Real> MatStiffness = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

        if (FEMType == 0)
        {
            TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();
            
            std::cout << triFEM->getRotatedInitialElement(0) << std::endl;
            std::cout << triFEM->getRotationMatrix(0) << std::endl;
            std::cout << triFEM->getMaterialStiffness(0) << std::endl;
            std::cout << triFEM->getStrainDisplacements(0) << std::endl;

            std::cout << triFEM->getRotatedInitialElement(1) << std::endl;
            std::cout << triFEM->getRotationMatrix(1) << std::endl;
            std::cout << triFEM->getMaterialStiffness(1) << std::endl;
            std::cout << triFEM->getStrainDisplacements(1) << std::endl;

        }
        else if (FEMType == 1)
        {
            TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
        }
    }



    void checkTriangleFEMValues()
    {
        // load Triangular FEM
        int nbrGrid = 40;
        createGridFEMScene(0, nbrGrid);

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
        createGridFEMScene(1, nbrGrid);

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


    void testTriangleFEMPerformance()
    {
        // init
        m_nbrStep = 1000;
        int nbrGrid = 40;

        // load Triangular FEM
        createGridFEMScene(0, nbrGrid);
        if (m_root.get() == nullptr)
            return;

        int nbrTest = 10;
        double diffTimeMs = 0;
        double timeMin = std::numeric_limits<double>::max();
        double timeMax = std::numeric_limits<double>::min();
        for (int i = 0; i < nbrTest; ++i)
        {
            ctime_t startTime = sofa::helper::system::thread::CTime::getRefTime();
            for (int i = 0; i < m_nbrStep; i++)
            {
                m_simulation->animate(m_root.get(), 0.01);
            }

            ctime_t diffTime = sofa::helper::system::thread::CTime::getRefTime() - startTime;
            double diffTimed = sofa::helper::system::thread::CTime::toSecond(diffTime);
            
            if (timeMin > diffTimed)
                timeMin = diffTimed;
            if (timeMax < diffTimed)
                timeMax = diffTimed;

            diffTimeMs += diffTimed;
            m_simulation->reset(m_root.get());
        }
        
        //std::cout << "timeMean: " << diffTimeMs/nbrTest << std::endl;
        //std::cout << "timeMin: " << timeMin << std::endl;
        //std::cout << "timeMax: " << timeMax << std::endl;

        //Record:
        //timeMean: 4.1545   ||   4.12943
        //timeMin : 4.07156  ||   4.05766
        //timeMax : 4.25747  ||   4.33603
    }

    void testTriangularFEMPerformance()
    {
        // init
        m_nbrStep = 1000;
        int nbrGrid = 40;

        // load Triangular FEM
        createGridFEMScene(1, nbrGrid);
        if (m_root.get() == nullptr)
            return;

        int nbrTest = 10;
        double diffTimeMs = 0;
        double timeMin = std::numeric_limits<double>::max();
        double timeMax = std::numeric_limits<double>::min();
        for (int i = 0; i < nbrTest; ++i)
        {
            ctime_t startTime = sofa::helper::system::thread::CTime::getRefTime();
            for (int i = 0; i < m_nbrStep; i++)
            {
                m_simulation->animate(m_root.get(), 0.01);
            }

            ctime_t diffTime = sofa::helper::system::thread::CTime::getRefTime() - startTime;
            double diffTimed = sofa::helper::system::thread::CTime::toSecond(diffTime);

            if (timeMin > diffTimed)
                timeMin = diffTimed;
            if (timeMax < diffTimed)
                timeMax = diffTimed;

            diffTimeMs += diffTimed;
            m_simulation->reset(m_root.get());
        }

        //std::cout << "timeMean: " << diffTimeMs / nbrTest << std::endl;
        //std::cout << "timeMin: " << timeMin << std::endl;
        //std::cout << "timeMax: " << timeMax << std::endl;

        //Record:
        //timeMean: 5.32281   ||  5.21513   ||  4.32919
        //timeMin : 5.171     ||  5.1868    ||  4.27747
        //timeMax : 5.50842   ||  5.29571   ||  4.3987

        // Optimisation:
        // addDForce: 0.00363014
        // addDForce: 0.00317752 -> getTriangle()[]
        // addDForce: 0.00290456 ->  const &

        // addForce: 0.0011347
        // addForce: 0.00110351
        // addForce: 0.00094206
    }

};


typedef TriangleFEMForceField_test<Vec3Types> TriangleFEMForceField3_test;

/// Tests for TriangleFEMForceField
TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_Creation)
{
    this->checkCreation(0);
}

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_noTopology)
{
    this->checkNoTopology(0);
}

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_emptyTopology)
{
    this->checkEmptyTopology(0);
}

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_defaultAttributes)
{
    this->checkDefaultAttributes(0);
}

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_wrongAttributess)
{
    this->checkWrongAttributes(0);
}

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_init)
{
    this->checkInit(0);
}


TEST_F(TriangleFEMForceField3_test, checkTriangleFEMValues)
{
    this->checkTriangleFEMValues();
}




/// Tests for TriangularFEMForceField  TODO: remove them when component has been fully merged into TriangleFEMForceField
TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_Creation)
{
    this->checkCreation(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_NoTopology)
{
    this->checkNoTopology(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_emptyTopology)
{
    this->checkEmptyTopology(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_defaultAttributes)
{
    this->checkDefaultAttributes(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_wrongAttributess)
{
    this->checkWrongAttributes(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMForceField_init)
{
    this->checkInit(1);
}

TEST_F(TriangleFEMForceField3_test, checkTriangularFEMValues)
{
    this->checkTriangularFEMValues();
}


/// Those tests should not be removed but can't be run on the CI
#if(PERFORMANCE_TESTS)
TEST_F(TriangleFEMForceField3_test, testTriangleFEMPerformance)
{
    this->testTriangleFEMPerformance();
}

TEST_F(TriangleFEMForceField3_test, testTriangularFEMPerformance)
{
    this->testTriangularFEMPerformance();
}
#endif
} // namespace sofa
