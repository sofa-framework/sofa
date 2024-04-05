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
#include <sofa/testing/BaseTest.h>
#include <sofa/component/linearsolver/direct/SparseLDLSolver.h>
#include <sofa/component/linearsolver/direct/SparseCommon.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/testing/NumericTest.h>


TEST(SparseLDLSolver, EmptySystem)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;

    MatrixType matrix;
    matrix.resize(0, 0);
    matrix.compress();

    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    solver->init();

    EXPECT_MSG_EMIT(Warning);
    solver->invert(matrix);
}

TEST(SparseLDLSolver, EmptyMState)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

    const sofa::simulation::Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("root");

    sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Direct");
    sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
    sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");

    sofa::simpleapi::createObject(root, "DefaultAnimationLoop");
    sofa::simpleapi::createObject(root, "EulerImplicitSolver");
    sofa::simpleapi::createObject(root, "SparseLDLSolver", {{"template", "CompressedRowSparseMatrixd"}});
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}, {"position", ""}});

    sofa::simulation::node::initRoot(root.get());

    {
        EXPECT_MSG_EMIT(Warning);
        sofa::simulation::node::animate(root.get(), 0.5_sreal);
    }

    sofa::simulation::node::unload(root);
}


// A topological change occurs leading to a mechanical object without any DOF
TEST(SparseLDLSolver, TopologyChangeEmptyMState)
{
    // required to be able to use EXPECT_MSG_NOEMIT and EXPECT_MSG_EMIT
    sofa::helper::logging::MessageDispatcher::addHandler(sofa::testing::MainGtestMessageHandler::getInstance() ) ;

  const sofa::simulation::Node::SPtr root = sofa::simulation::getSimulation()->createNewGraph("root");

    sofa::simpleapi::importPlugin("Sofa.Component.LinearSolver.Direct");
    sofa::simpleapi::importPlugin("Sofa.Component.Mass");
    sofa::simpleapi::importPlugin("Sofa.Component.ODESolver.Backward");
    sofa::simpleapi::importPlugin("Sofa.Component.StateContainer");
    sofa::simpleapi::importPlugin("Sofa.Component.Topology.Container.Dynamic");
    sofa::simpleapi::importPlugin("Sofa.Component.Topology.Utility");

    sofa::simpleapi::createObject(root, "DefaultAnimationLoop");
    sofa::simpleapi::createObject(root, "EulerImplicitSolver");
    sofa::simpleapi::createObject(root, "SparseLDLSolver", {{"template", "CompressedRowSparseMatrixd"}});
    sofa::simpleapi::createObject(root, "PointSetTopologyContainer", {{"position", "0 0 0"}});
    sofa::simpleapi::createObject(root, "PointSetTopologyModifier");
    sofa::simpleapi::createObject(root, "MechanicalObject", {{"template", "Vec3"}});
    sofa::simpleapi::createObject(root, "UniformMass", {{"totalMass", "1.0"}});
    sofa::simpleapi::createObject(root, "TopologicalChangeProcessor",
                                  {{"useDataInputs", "true"}, {"timeToRemove", "0.05"},
                                   {"pointsToRemove", "0"}});

    sofa::simulation::node::initRoot(root.get());

    {
        EXPECT_MSG_NOEMIT(Warning);
        sofa::simulation::node::animate(root.get(), 0.1_sreal);
    }

    {
        EXPECT_MSG_EMIT(Warning);
        sofa::simulation::node::animate(root.get(), 0.1_sreal);
    }

    sofa::simulation::node::unload(root);
}



TEST(SparseLDLSolver, AssociatedLinearSystem)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    solver->init();
    EXPECT_NE(solver->getContext(), nullptr);

    auto* system = solver->getLinearSystem();
    EXPECT_NE(system, nullptr);

    using MatrixSystem = sofa::component::linearsystem::MatrixLinearSystem<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    auto* matrixSystem = dynamic_cast<MatrixSystem*>(system);
    EXPECT_NE(matrixSystem, nullptr);

    EXPECT_EQ(MatrixSystem::GetCustomTemplateName(), MatrixType::Name());
}
