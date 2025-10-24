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
#pragma once

#include <sofa/component/statecontainer/MechanicalObject.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/testing/NumericTest.h>
#include <sofa/testing/TestMessageHandler.h>


namespace sofa
{
using sofa::type::Vec3;
using sofa::type::Vec6;

template <class _TetrahedronFEMForceField>
class BaseTetrahedronFEMForceField_test : public sofa::testing::BaseTest
{
public:
    using DataTypes = typename _TetrahedronFEMForceField::DataTypes;

    using Real = typename DataTypes::Real;
    using Coord = typename DataTypes::Coord;
    using VecCoord = typename DataTypes::VecCoord;

    using MState = component::statecontainer::MechanicalObject<DataTypes>;

    using Transformation = type::Mat<3, 3, Real>;
    using MaterialStiffness = type::Mat<6, 6, Real>;
    using StrainDisplacement = type::Mat<12, 6, Real>;
    using TetraCoord = type::fixed_array<Coord, 4>;

    static constexpr const char* dataTypeName = DataTypes::Name();
    static inline std::string className = _TetrahedronFEMForceField::GetClass()->className;

protected:
    simulation::Simulation* m_simulation = nullptr;
    simulation::Node::SPtr m_root;

    helper::system::thread::ctime_t timeTicks = sofa::helper::system::thread::CTime::getRefTicksPerSec();

public:

    void doSetUp() override
    {
        m_simulation = sofa::simulation::getSimulation();
    }

    void doTearDown() override
    {
        if (m_root != nullptr)
            sofa::simulation::node::unload(m_root);
    }

    void addTetraFEMForceField(simulation::Node::SPtr node, Real young, Real poisson, std::string method)
    {
        sofa::simpleapi::createObject(node, className, {
            {"name","FEM"},
            {"youngModulus", sofa::simpleapi::str(young)},
            {"poissonRatio", sofa::simpleapi::str(poisson)},
            {"method", method}
        });
    }


    void createSingleTetrahedronFEMScene(Real young, Real poisson, std::string method)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");

        simpleapi::createObject(m_root, "DefaultAnimationLoop");
        simpleapi::createObject(m_root, "DefaultVisualManagerLoop");

        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.SolidMechanics.FEM.Elastic,
            Sofa.Component.Mass
        });

        simpleapi::createObject(m_root, "MechanicalObject", { {"template", dataTypeName}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        simpleapi::createObject(m_root, "TetrahedronSetTopologyContainer", { {"tetrahedra","2 3 1 0"} });
        simpleapi::createObject(m_root, "TetrahedronSetTopologyModifier");
        simpleapi::createObject(m_root, "TetrahedronSetGeometryAlgorithms", { {"template", dataTypeName} });

        addTetraFEMForceField(m_root, young, poisson, method);

        simpleapi::createObject(m_root, "DiagonalMass", {
            {"name","mass"}, {"massDensity","0.1"} });
        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }


    void createGridFEMScene(type::Vec3 nbrGrid)
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        m_root->setGravity(type::Vec3(0.0, 10.0, 0.0));
        m_root->setDt(0.01);

        simpleapi::createObject(m_root, "DefaultAnimationLoop");
        simpleapi::createObject(m_root, "DefaultVisualManagerLoop");

        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.Topology.Container.Grid,
            Sofa.Component.SolidMechanics.FEM.Elastic,
            Sofa.Component.Mass,
            Sofa.Component.LinearSolver.Iterative,
            Sofa.Component.ODESolver.Backward,
            Sofa.Component.Constraint.Lagrangian,
            Sofa.Component.Topology.Mapping,
            Sofa.Component.Engine.Select,
            Sofa.Component.Constraint.Projective
        });

        simpleapi::createObject(m_root, "ProjectedGaussSeidelConstraintSolver", { {"tolerance", "1e-3"}, {"maxIt", "1000"} });

        simpleapi::createObject(m_root, "RegularGridTopology", { {"name", "grid"},
                    {"n", sofa::simpleapi::str(nbrGrid)}, {"min", "0 0 20"}, {"max", "10 40 30"} });


        simulation::Node::SPtr FEMNode = sofa::simpleapi::createChild(m_root, "Beam");
        simpleapi::createObject(FEMNode, "EulerImplicitSolver");
        //createObject(FEMNode, "SparseLDLSolver", { {"name","solver"}, { "template", "CompressedRowSparseMatrixd" } });
        simpleapi::createObject(FEMNode, "CGLinearSolver", { { "iterations", "20" }, { "tolerance", "1e-5" }, {"threshold", "1e-6"} });

        simpleapi::createObject(FEMNode, "MechanicalObject", {
            {"name","dof"}, {"template",dataTypeName}, {"position", "@../grid.position"} });

        simpleapi::createObject(FEMNode, "TetrahedronSetTopologyContainer", {
            {"name","topo"} });
        simpleapi::createObject(FEMNode, "TetrahedronSetTopologyModifier", {
            {"name","Modifier"} });
        simpleapi::createObject(FEMNode, "TetrahedronSetGeometryAlgorithms", {
            {"name","GeomAlgo"}, {"template",dataTypeName} });
        simpleapi::createObject(FEMNode, "Hexa2TetraTopologicalMapping", {
            {"input","@../grid"}, {"output","@topo"} });

        simpleapi::createObject(FEMNode, "BoxROI", {
            {"name","ROI1"}, {"box","-1 -1 0 10 1 50"} });
        simpleapi::createObject(FEMNode, "FixedProjectiveConstraint", { {"mstate", "@dof"},
            {"name","fixC"}, {"indices","@ROI1.indices"} });

        simpleapi::createObject(FEMNode, "DiagonalMass", {
            {"name","mass"}, {"massDensity","1.0"} });

        addTetraFEMForceField(FEMNode, 600, 0.3, "large");

        ASSERT_NE(m_root.get(), nullptr);

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }



    void checkCreation()
    {
        createSingleTetrahedronFEMScene(static_cast<Real>(10000), static_cast<Real>(0.4), "large");

        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);
        ASSERT_EQ(dofs->getSize(), 4);

        typename _TetrahedronFEMForceField::SPtr tetraFEM = m_root->getTreeObject<_TetrahedronFEMForceField>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);
        ASSERT_FLOATINGPOINT_EQ(tetraFEM->d_poissonRatio.getValue()[0], static_cast<Real>(0.4));
        ASSERT_FLOATINGPOINT_EQ(tetraFEM->d_youngModulus.getValue()[0], static_cast<Real>(10000));
        ASSERT_EQ(tetraFEM->d_method.getValue(), "large");
    }

    void checkNoTopology()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        simpleapi::createObject(m_root, "DefaultAnimationLoop");
        simpleapi::createObject(m_root, "DefaultVisualManagerLoop");

        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.SolidMechanics.FEM.Elastic
        });

        simpleapi::createObject(m_root, "MechanicalObject", { {"template","Vec3"}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        addTetraFEMForceField(m_root, 100, 0.3, "large");

        EXPECT_MSG_EMIT(Error);

        /// Init simulation
        sofa::simulation::node::initRoot(m_root.get());
    }

    virtual sofa::helper::logging::Message::Type expectedMessageWhenEmptyTopology() const { return sofa::helper::logging::Message::Error; }

    void checkEmptyTopology()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        simpleapi::createObject(m_root, "DefaultAnimationLoop");
        simpleapi::createObject(m_root, "DefaultVisualManagerLoop");

        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.SolidMechanics.FEM.Elastic
        });

        simpleapi::createObject(m_root, "MechanicalObject", { {"template","Vec3"} });
        simpleapi::createObject(m_root, "TetrahedronSetTopologyContainer");
        addTetraFEMForceField(m_root, 100, 0.3, "large");

        {
            sofa::testing::ExpectMessage failure(expectedMessageWhenEmptyTopology(), __FILE__, __LINE__);
            sofa::simulation::node::initRoot(m_root.get());
        }
    }


    void checkDefaultAttributes()
    {
        m_root = sofa::simpleapi::createRootNode(m_simulation, "root");
        simpleapi::createObject(m_root, "DefaultAnimationLoop");
        simpleapi::createObject(m_root, "DefaultVisualManagerLoop");

        this->loadPlugins({
            Sofa.Component.StateContainer,
            Sofa.Component.Topology.Container.Dynamic,
            Sofa.Component.SolidMechanics.FEM.Elastic
        });

        simpleapi::createObject(m_root, "MechanicalObject", { {"template", dataTypeName}, {"position", "0 0 0  1 0 0  0 1 0  0 0 1"} });
        simpleapi::createObject(m_root, "TetrahedronSetTopologyContainer", { {"tetrahedra","2 3 1 0"} });
        simpleapi::createObject(m_root, "TetrahedronSetTopologyModifier");
        simpleapi::createObject(m_root, "TetrahedronSetGeometryAlgorithms", { {"template", dataTypeName} });

        simpleapi::createObject(m_root, className);

        {
            EXPECT_MSG_EMIT(Error);
            sofa::simulation::node::initRoot(m_root.get());
        }

        typename _TetrahedronFEMForceField::SPtr tetraFEM = m_root->getTreeObject<_TetrahedronFEMForceField>();
        ASSERT_TRUE(tetraFEM.get() != nullptr);

        ASSERT_FLOATINGPOINT_EQ(tetraFEM->d_poissonRatio.getValue(), static_cast<Real>(0.45));
        ASSERT_FLOATINGPOINT_EQ(tetraFEM->d_youngModulus.getValue()[0], static_cast<Real>(5000));
        ASSERT_EQ(tetraFEM->d_method.getValue(), "large");
    }


    void checkWrongAttributes()
    {
        EXPECT_MSG_EMIT(Warning);
        createSingleTetrahedronFEMScene(-100, -0.3, "toto");
    }

    virtual void computeMatricesCheckInit(Transformation& initRot, Transformation& curRot, MaterialStiffness& stiffnessMat, StrainDisplacement& strainD, TetraCoord& initPosition, sofa::Size elementId)
    {
        SOFA_UNUSED(initRot);
        SOFA_UNUSED(curRot);
        SOFA_UNUSED(stiffnessMat);
        SOFA_UNUSED(strainD);
        SOFA_UNUSED(initPosition);
        SOFA_UNUSED(elementId);
    }

    void setupCheckInit(Transformation& exp_initRot, TetraCoord& exp_initPos, Transformation& exp_curRot, MaterialStiffness& exp_stiffnessMat, StrainDisplacement& exp_strainD)
    {
        createSingleTetrahedronFEMScene(static_cast<Real>(1000), static_cast<Real>(0.3), "large");

        // Expected values
        exp_initRot = { Vec3(0, 0.816497, 0.57735), Vec3(-0.707107, -0.408248, 0.57735), Vec3(0.707107, -0.408248, 0.57735) };
        exp_initPos = { Coord(0, 0, 0), Coord(1.41421, 0, 0), Coord(0.707107, 1.22474, 0), Coord(0.707107, 0.408248, -0.57735) };

        exp_curRot = { Vec3(0, 0.816497, 0.57735), Vec3(-0.707107, -0.408248, 0.57735), Vec3(0.707107, -0.408248, 0.57735) };

        exp_stiffnessMat = { Vec6(224.359, 96.1538, 96.1538, 0, 0, 0), Vec6(96.1538, 224.359, 96.1538, 0, 0, 0), Vec6(96.1538, 96.1538, 224.359, 0, 0, 0),
            Vec6(0, 0, 0, 64.1026, 0, 0), Vec6(0, 0, 0, 0, 64.1026, 0), Vec6(0, 0, 0, 0, 0, 64.1026) };

        exp_strainD = { Vec6(0.707107, 0, 0, 0.408248, 0, -0.57735),
            Vec6(0, 0.408248, 0, 0.707107, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, 0.408248, 0.707107),
            Vec6(-0.707107, 0, 0, 0.408248, 0, -0.57735),
            Vec6(0, 0.408248, 0, -0.707107, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, 0.408248, -0.707107),
            Vec6(-0, 0, 0, -0.816497, 0, -0.57735),
            Vec6(0, -0.816497, 0, -0, -0.57735, 0),
            Vec6(0, 0, -0.57735, 0, -0.816497, 0),
            Vec6(0, 0, 0, -0, 0, 1.73205),
            Vec6(0, 0, 0, 0, 1.73205, 0),
            Vec6(0, 0, 1.73205, 0, 0, 0) };
    }

    virtual void checkInit()
    {
        Transformation exp_initRot;
        TetraCoord exp_initPos;
        Transformation exp_curRot;
        MaterialStiffness exp_stiffnessMat;
        StrainDisplacement exp_strainD;

        setupCheckInit(exp_initRot, exp_initPos, exp_curRot, exp_stiffnessMat, exp_strainD);

        Transformation initRot (type::NOINIT);
        Transformation curRot(type::NOINIT);
        MaterialStiffness stiffnessMat(type::NOINIT);
        StrainDisplacement strainD(type::NOINIT);
        TetraCoord initPosition;

        computeMatricesCheckInit(initRot, curRot, stiffnessMat, strainD, initPosition, 0);

        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check position
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initPos[i][j], initPosition[i][j], 1e-4);
            }
        }

        // check stiffness
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_stiffnessMat[i][j], stiffnessMat[i][j], 1e-4);
            }
        }

        // check strain displacement
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_strainD[i][j], strainD[i][j], 1e-4);
            }
        }

    }

    virtual void computeMatricesCheckFEMValues(Transformation& initRot, Transformation& curRot, MaterialStiffness& stiffnessMat, StrainDisplacement& strainD, TetraCoord& initPosition, sofa::Size elementId)
    {
        computeMatricesCheckInit(initRot, curRot, stiffnessMat, strainD, initPosition, elementId);
    }

    void setupCheckFEMValues(Transformation& exp_initRot, TetraCoord& exp_initPos, Transformation& exp_curRot, MaterialStiffness& exp_stiffnessMat, StrainDisplacement& exp_strainD)
    {
        type::Vec3 grid = type::Vec3(4, 10, 4);

        // load TetrahedronFEMForceField grid
        createGridFEMScene(grid);
        if (m_root.get() == nullptr)
            return;

        // Access mstate
        typename MState::SPtr dofs = m_root->getTreeObject<MState>();
        ASSERT_TRUE(dofs.get() != nullptr);

        // Access dofs
        const VecCoord& positions = dofs->x.getValue();
        ASSERT_EQ(positions.size(), 4 * 10 * 4);

        EXPECT_NEAR(positions[159][0], 10, 1e-4);
        EXPECT_NEAR(positions[159][1], 40, 1e-4);
        EXPECT_NEAR(positions[159][2], 30, 1e-4);

        // perform some steps
        for (int i = 0; i < 100; i++)
        {
            sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
        }

        EXPECT_NEAR(positions[159][0], 9.99985, 1e-4);
        EXPECT_NEAR(positions[159][1], 45.0487, 1e-4);
        EXPECT_NEAR(positions[159][2], 30.0011, 1e-4);

        // Expected values
        exp_initRot = { Vec3(-1, 0, 0), Vec3(0, -0.8, -0.6), Vec3(0, -0.6, 0.8) };
        exp_initPos = { Coord(0, 0, 0), Coord(3.33333, 0, 0), Coord(3.33333, 5.55556, 0), Coord(0, 3.55556, 2.66667) };

        exp_curRot = { Vec3(-1, 8.01488e-06, 0.000541687), Vec3(-0.000320764, -0.814541, -0.580106), Vec3(0.000436576, -0.580106, 0.814541) };

        exp_stiffnessMat = { Vec6(2.72596, 1.16827, 1.16827, 0, 0, 0), Vec6(1.16827, 2.72596, 1.16827, 0, 0, 0), Vec6(1.16827, 1.16827, 2.72596, 0, 0, 0),
            Vec6(0, 0, 0, 0.778846, 0, 0), Vec6(0, 0, 0, 0, 0.778846, 0), Vec6(0, 0, 0, 0, 0, 0.778846) };

        exp_strainD = { Vec6(-14.8148, 0, 0, 1.18424e-14, 0, -18.5185),
            Vec6(0, 1.18424e-14, 0, -14.8148, -18.5185, 0),
            Vec6(0, 0, -18.5185, 0, 1.18424e-14, -14.8148),
            Vec6(14.8148, 0, 0, -8.88889, 0, 11.8519),
            Vec6(0, -8.88889, 0, 14.8148, 11.8519, 0),
            Vec6(0, 0, 11.8519, 0, -8.88889, 14.8148),
            Vec6(-0, 0, 0, 8.88889, 0, -11.8519),
            Vec6(0, 8.88889, 0, -0, -11.8519, 0),
            Vec6(0, 0, -11.8519, 0, 8.88889, -0),
            Vec6(0, 0, 0, -1.18424e-14, 0, 18.5185),
            Vec6(0, -1.18424e-14, 0, 0, 18.5185, 0),
            Vec6(0, 0, 18.5185, 0, -1.18424e-14, 0) };
    }

    virtual void checkFEMValues()
    {
        Transformation exp_initRot;
        TetraCoord exp_initPos;
        Transformation exp_curRot;
        MaterialStiffness exp_stiffnessMat;
        StrainDisplacement exp_strainD;

        setupCheckFEMValues(exp_initRot, exp_initPos, exp_curRot, exp_stiffnessMat, exp_strainD);

        Transformation initRot(type::NOINIT);
        Transformation curRot(type::NOINIT);
        MaterialStiffness stiffnessMat(type::NOINIT);
        StrainDisplacement strainD(type::NOINIT);
        TetraCoord initPosition;

        constexpr sofa::Size elementId = 100;
        computeMatricesCheckFEMValues(initRot, curRot, stiffnessMat, strainD, initPosition, elementId);

        // check rotations
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initRot[i][j], initRot[i][j], 1e-4);
                EXPECT_NEAR(exp_curRot[i][j], curRot[i][j], 1e-4);
            }
        }

        // check position
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                EXPECT_NEAR(exp_initPos[i][j], initPosition[i][j], 1e-4);
            }
        }

        // check stiffness
        for (int i = 0; i < 6; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_stiffnessMat[i][j], stiffnessMat[i][j], 1e-4);
            }
        }

        // check strain displacement
        for (int i = 0; i < 12; ++i)
        {
            for (int j = 0; j < 6; ++j)
            {
                EXPECT_NEAR(exp_strainD[i][j], strainD[i][j], 1e-4);
            }
        }
    }


    void testFEMPerformance()
    {
        const type::Vec3 grid = type::Vec3(8, 26, 8);

        // load TetrahedronFEMForceField grid
        createGridFEMScene(grid);
        if (m_root.get() == nullptr)
            return;

        const int nbrStep = 1000;
        const int nbrTest = 4;
        double diffTimeMs = 0;
        double timeMin = std::numeric_limits<double>::max();
        double timeMax = std::numeric_limits<double>::min();

        if (m_simulation == nullptr)
            return;

        for (int i = 0; i < nbrTest; ++i)
        {
            const helper::system::thread::ctime_t startTime = sofa::helper::system::thread::CTime::getRefTime();
            for (int i = 0; i < nbrStep; i++)
            {
                sofa::simulation::node::animate(m_root.get(), 0.01_sreal);
            }

            const helper::system::thread::ctime_t diffTime = sofa::helper::system::thread::CTime::getRefTime() - startTime;
            const double diffTimed = sofa::helper::system::thread::CTime::toSecond(diffTime);

            if (timeMin > diffTimed)
                timeMin = diffTimed;
            if (timeMax < diffTimed)
                timeMax = diffTimed;

            diffTimeMs += diffTimed;
            sofa::simulation::node::reset(m_root.get());
        }

        std::cout << "timeMean: " << diffTimeMs / nbrTest << std::endl;
        std::cout << "timeMin: " << timeMin << std::endl;
        std::cout << "timeMax: " << timeMax << std::endl;

        //TetrahedronFEM
        //timeMean: 7.40746
        //timeMin : 7.37514
        //timeMax : 7.46645

        //TetrahedralCorotational
        //timeMean : 14.0486
        //timeMin : 13.9016
        //timeMax : 14.4603

        // FastTetrahedralCorotationalForceField
        //timeMean: 6.01042
        //timeMin : 5.95179
        //timeMax : 6.16263
    }


};


TYPED_TEST_SUITE_P(BaseTetrahedronFEMForceField_test);

TYPED_TEST_P(BaseTetrahedronFEMForceField_test, creation)
{
    this->checkCreation();
}

TYPED_TEST_P(BaseTetrahedronFEMForceField_test, noTopology)
{
    this->checkNoTopology();
}

// TYPED_TEST_P(BaseTetrahedronFEMForceField_test, emptyTopology)
// {
//     this->checkEmptyTopology();
// }

TYPED_TEST_P(BaseTetrahedronFEMForceField_test, DISABLED_testFEMPerformance)
{
    this->testFEMPerformance();
}

REGISTER_TYPED_TEST_SUITE_P(BaseTetrahedronFEMForceField_test,
                            creation, noTopology, DISABLED_testFEMPerformance);

}
