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
    using Vec3 = type::Vec<3, Real>;
    using Mat33 = type::Mat<3, 3, Real>;
    using Mat63 = type::Mat<6, 3, Real>;

protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;

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
            fixP = static_cast<unsigned int>(nbrGrid - 1);
        
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
        createObject(FEMNode, "CGLinearSolver", {{ "iterations", "20" }, { "threshold", "1e-6" }});

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

        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        if (FEMType == 0)
        {
            typename TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3);
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 100);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
        else if (FEMType == 1)
        {
            typename TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
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

        EXPECT_MSG_EMIT(Warning);

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

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::getSimulation()->init(m_root.get());
        if (FEMType == 0)
        {
            typename TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3);
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 1000);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
        else if (FEMType == 1)
        {
            typename TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
            ASSERT_TRUE(triFEM.get() != nullptr);
            ASSERT_FLOAT_EQ(triFEM->getPoisson(), 0.3); // Not the same default values
            ASSERT_FLOAT_EQ(triFEM->getYoung(), 1000);
            ASSERT_EQ(triFEM->getMethod(), 0);
        }
    }


    void checkWrongAttributes(int FEMType)
    {
        EXPECT_MSG_EMIT(Warning);
        createSingleTriangleFEMScene(FEMType, -100, -0.3, "toto");
    }


    void checkInit(int FEMType)
    {
        createSingleTriangleFEMScene(FEMType, 100, 0.3, "large");
        
        type::Vec<2, Mat33> exp_rotInit;
        type::Vec<2, Mat33> exp_rotMat;
        type::Vec<2, Mat33> exp_stiffnessMat;
        type::Vec<2, Mat63> exp_strainDispl;

        // 1st value expected values (square 2D triangle)
        exp_rotInit[0] = Mat33(Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(0, 1, 0));
        exp_rotMat[0] = Mat33(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec3(0, 0, 1));
        exp_stiffnessMat[0] = Mat33(Vec3(54.945053, 16.483517, 0), Vec3(16.483517, 54.945053, 0), Vec3(0, 0, 19.23077));
        exp_strainDispl[0][0] = Vec3(-1, 0, -1); exp_strainDispl[0][1] = Vec3(0, -1, -1); exp_strainDispl[0][2] = Vec3(1, 0, 0);
        exp_strainDispl[0][3] = Vec3(0, 0, 1); exp_strainDispl[0][4] = Vec3(0, 0, 1); exp_strainDispl[0][5] = Vec3(0, 1, 0);

        // 2nd value expected values (isosceles 3D triangle)
        exp_rotInit[1] = Mat33(Vec3(0, 0, 0), Vec3(1.4142135, 0, 0), Vec3(0.707107, 1.2247449, 0));
        exp_rotMat[1] = Mat33(Vec3(0, -0.81649661, -0.57735), Vec3(0.707107, 0.40824831, -0.57735), Vec3(0.707107, -0.40824831, 0.57735));
        exp_stiffnessMat[1] = Mat33(Vec3(95.1676, 28.550287, 0), Vec3(28.550287, 95.1676, 0), Vec3(0, 0, 33.30867));
        exp_strainDispl[1][0] = Vec3(-1, 0, -1); exp_strainDispl[1][1] = Vec3(0, -1, -1); exp_strainDispl[1][2] = Vec3(1, 0, 0);
        exp_strainDispl[1][3] = Vec3(0, 0, 1); exp_strainDispl[1][4] = Vec3(0, 0, 1); exp_strainDispl[1][5] = Vec3(0, 1, 0);

        if (FEMType == 0)
        {
            typename TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();

            for (int id = 0; id < 2; id++)
            {
                const type::fixed_array <Coord, 3>& rotInit = triFEM->getRotatedInitialElement(id);
                const Mat33& rotMat = triFEM->getRotationMatrix(id);
                const Mat33& stiffnessMat = triFEM->getMaterialStiffness(id);
                const Mat63& strainDispl = triFEM->getStrainDisplacements(id);

                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        EXPECT_NEAR(rotInit[i][j], exp_rotInit[id][i][j], 1e-4);
                        EXPECT_NEAR(rotMat[i][j], exp_rotMat[id][i][j], 1e-4);
                        EXPECT_NEAR(stiffnessMat[i][j], exp_stiffnessMat[id][i][j], 1e-4);

                        EXPECT_NEAR(strainDispl[i][j], exp_strainDispl[id][i][j], 1e-4);
                        EXPECT_NEAR(strainDispl[i + 3][j], exp_strainDispl[id][i + 3][j], 1e-4);
                    }
                }
            }
        }
        else if (FEMType == 1)
        {
            typename TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
            for (int id = 0; id < 2; id++)
            {
                typename TriangularFEM::TriangleInformation triangleInfo = triFEM->triangleInfo.getValue()[id];
                const type::fixed_array <Coord, 3>& rotInit = triangleInfo.rotatedInitialElements;
                const Mat33& rotMat = triangleInfo.initialTransformation;
                const Mat33& stiffnessMat = triangleInfo.materialMatrix;
                const Mat63& strainDispl = triangleInfo.strainDisplacementMatrix;

                for (int i = 0; i < 3; ++i)
                {
                    for (int j = 0; j < 3; ++j)
                    {
                        EXPECT_NEAR(rotInit[i][j], exp_rotInit[id][i][j], 1e-4);
                        EXPECT_NEAR(rotMat[i][j], exp_rotMat[id][i][j], 1e-4);
                        EXPECT_NEAR(stiffnessMat[i][j], exp_stiffnessMat[id][i][j], 1e-4);

                        EXPECT_NEAR(strainDispl[i][j], exp_strainDispl[id][i][j], 1e-4);
                        EXPECT_NEAR(strainDispl[i + 3][j], exp_strainDispl[id][i + 3][j], 1e-4);
                    }
                }
            }
        }
    }



    void checkFEMValues(int FEMType)
    {
        // load Triangular FEM
        int nbrGrid = 40;
        int nbrStep = 100;
        createGridFEMScene(FEMType, nbrGrid);

        if (m_root.get() == nullptr)
            return;

        // Access mstate
        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        
        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), nbrGrid * nbrGrid);
        
        EXPECT_NEAR(positions[1515][0], 8.97436, 1e-4);
        EXPECT_NEAR(positions[1515][1], 9.48718, 1e-4);
        EXPECT_NEAR(positions[1515][2], 0, 1e-4);

        for (int i = 0; i < nbrStep; i++)
        {
            m_simulation->animate(m_root.get(), 0.01);
        }

        EXPECT_NEAR(positions[1515][0], 8.9135, 1e-4);
        EXPECT_NEAR(positions[1515][1], 14.2499, 1e-4);
        EXPECT_NEAR(positions[1515][2], 0, 1e-4);

        // 1st value expected values (square 2D triangle)
        static const Mat33 exp_rotInit = Mat33(Vec3(0, 0, 0), Vec3(0.25641, 0, 0), Vec3(0.25641, 0.25641, 0));
        static const Mat33 exp_rotMat = Mat33(Vec3(0.99992, -0.0126608, 0), Vec3(0.0126608, 0.99992, 0), Vec3(0, 0, 1));
        static const Mat33 exp_stiffnessMat = Mat33(Vec3(3.61243, 1.08373, 0), Vec3(1.08373, 3.61243, 0), Vec3(0, 0, 1.26435));
        Mat63 exp_strainDispl;
        exp_strainDispl[0] = Vec3(-3.89456, 0, -0.00185328); exp_strainDispl[1] = Vec3(0, -0.00185328, -3.89456); exp_strainDispl[2] = Vec3(3.89456, 0, -3.89816);
        exp_strainDispl[3] = Vec3(0, -3.89816, 3.89456); exp_strainDispl[4] = Vec3(0, 0, 3.90001); exp_strainDispl[5] = Vec3(0, 3.90001, 0);

        if (FEMType == 0)
        {
            typename TriangleFEM::SPtr triFEM = m_root->getTreeObject<TriangleFEM>();

            const type::fixed_array <Coord, 3>& rotInit = triFEM->getRotatedInitialElement(42);
            const Mat33& rotMat = triFEM->getRotationMatrix(42);
            const Mat33& stiffnessMat = triFEM->getMaterialStiffness(42);
            const Mat63& strainDispl = triFEM->getStrainDisplacements(42);

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(rotInit[i][j], exp_rotInit[i][j], 1e-4);
                    EXPECT_NEAR(rotMat[i][j], exp_rotMat[i][j], 1e-4);
                    EXPECT_NEAR(stiffnessMat[i][j], exp_stiffnessMat[i][j], 1e-4);

                    EXPECT_NEAR(strainDispl[i][j], exp_strainDispl[i][j], 1e-4);
                    EXPECT_NEAR(strainDispl[i + 3][j], exp_strainDispl[i + 3][j], 1e-4);
                }
            }
        }
        else if (FEMType == 1)
        {
            typename TriangularFEM::SPtr triFEM = m_root->getTreeObject<TriangularFEM>();
            
            typename TriangularFEM::TriangleInformation triangleInfo = triFEM->triangleInfo.getValue()[42];
            const type::fixed_array <Coord, 3>& rotInit = triangleInfo.rotatedInitialElements;
            const Mat33& rotMat = triangleInfo.initialTransformation; // rotMat: [1 0 0,0 1 0,0 0 1]
            const Mat33& stiffnessMat = triangleInfo.materialMatrix;
            const Mat63& strainDispl = triangleInfo.strainDisplacementMatrix;

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    EXPECT_NEAR(rotInit[i][j], exp_rotInit[i][j], 1e-4);
                    EXPECT_NEAR(rotMat[i][j], exp_rotMat[i][j], 1e-4);
                    EXPECT_NEAR(stiffnessMat[i][j], exp_stiffnessMat[i][j], 1e-4);

                    EXPECT_NEAR(strainDispl[i][j], exp_strainDispl[i][j], 1e-4);
                    EXPECT_NEAR(strainDispl[i + 3][j], exp_strainDispl[i + 3][j], 1e-4);
                }
            }
        }
    }


    void testFEMPerformance(int FEMType)
    {
        // init
        int nbrStep = 1000;
        int nbrGrid = 40;

        // load Triangular FEM
        createGridFEMScene(FEMType, nbrGrid);
        if (m_root.get() == nullptr)
            return;

        int nbrTest = 10;
        double diffTimeMs = 0;
        double timeMin = std::numeric_limits<double>::max();
        double timeMax = std::numeric_limits<double>::min();
        for (int i = 0; i < nbrTest; ++i)
        {
            ctime_t startTime = sofa::helper::system::thread::CTime::getRefTime();
            for (int i = 0; i < nbrStep; i++)
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

        // ### Record values ###
        // {
        //// TriangleFEMModel
        //  timeMean: 4.1545   ||   4.12943
        //  timeMin : 4.07156  ||   4.05766
        //  timeMax : 4.25747  ||   4.33603
        // 
        //// TriangularFEMModel                 after optimisation
        //  timeMean: 5.32281 || 5.21513 || 4.32919
        //  timeMin : 5.171     ||  5.1868    ||  4.27747
        //  timeMax : 5.50842   ||  5.29571   ||  4.3987
        // Optimisation:
        // addDForce: 0.00363014
        // addDForce: 0.00317752 -> getTriangle()[]
        // addDForce: 0.00290456 ->  const &

        // addForce: 0.0011347
        // addForce: 0.00110351
        // addForce: 0.00094206
        //}
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

TEST_F(TriangleFEMForceField3_test, checkTriangleFEMForceField_values)
{
    this->checkFEMValues(0);
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

TEST_F(TriangleFEMForceField3_test, DISABLED_checkTriangularFEMForceField_init)
{
    this->checkInit(1);
}

TEST_F(TriangleFEMForceField3_test, DISABLED_checkTriangularFEMForceField_values)
{
    this->checkFEMValues(1);
}


/// Those tests should not be removed but can't be run on the CI
TEST_F(TriangleFEMForceField3_test, DISABLED_testTriangleFEMPerformance)
{
    this->testFEMPerformance(0);
}

TEST_F(TriangleFEMForceField3_test, DISABLED_testTriangularFEMPerformance)
{
    this->testFEMPerformance(1);
}

} // namespace sofa
