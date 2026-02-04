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
#include <sofa/component/linearsolver/direct/SparseCommon.h>
#include <sofa/component/linearsolver/direct/SparseLDLSolver.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/helper/RandomGenerator.h>
#include <sofa/simpleapi/SimpleApi.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/testing/BaseTest.h>
#include <sofa/testing/NumericTest.h>

#include "sofa/type/MatSym.h"

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

    const auto plugins = sofa::testing::makeScopedPlugin({
        Sofa.Component.LinearSolver.Direct,
        Sofa.Component.ODESolver.Backward,
        Sofa.Component.StateContainer});

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

    const auto plugins = sofa::testing::makeScopedPlugin({
        Sofa.Component.LinearSolver.Direct,
        Sofa.Component.Mass,
        Sofa.Component.ODESolver.Backward,
        Sofa.Component.StateContainer,
        Sofa.Component.Topology.Container.Dynamic,
        Sofa.Component.Topology.Utility});

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


TEST(SparseLDLSolver, TestInvertingRandomMatrix)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, sofa::linearalgebra::FullVector<SReal> >;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    solver->init();

    unsigned nbRows = 300;
    unsigned nbCols = 300;
    SReal reg = 5;
    float sparsity = 0.05;
    const auto nbNonZero = static_cast<sofa::SignedIndex>(sparsity * static_cast<SReal>(nbRows*nbCols));


    sofa::linearalgebra::FullMatrix<SReal> tempMatrix(nbRows,nbCols), finalTempMatrix;
    tempMatrix.clear();

    sofa::helper::RandomGenerator randomGenerator;
    randomGenerator.initSeed(2807);


    for (sofa::SignedIndex i = 0; i < nbNonZero; ++i)
    {
        const auto value = static_cast<SReal>(fabs(sofa::helper::drand(2)));
        const auto row = randomGenerator.random<sofa::Index>(0, nbRows);
        const auto col = randomGenerator.random<sofa::Index>(0, nbCols);
        tempMatrix.set(row,col,value);
    }

    for (sofa::SignedIndex i = 0; i < nbRows; ++i)
    {
        tempMatrix.set(i,i,tempMatrix(i,i) + reg);
    }
    tempMatrix.mulT(finalTempMatrix, tempMatrix);

    sofa::linearalgebra::CompressedRowSparseMatrix<SReal> matrix;
    matrix.resize(nbRows, nbCols);

    unsigned nbNZ = 0;
    for (unsigned i=0; i<nbRows; ++i)
    {
        for (unsigned j=0; j<nbCols; ++j)
        {
            if (finalTempMatrix(i,j) > 1e-8)
                matrix.set(i,j,finalTempMatrix(i,j));
            else
                ++nbNZ;
        }

    }
    msg_info("TestInvertingRandomMatrix") << "REAL SPARSITY (#zeros/#elements) : " <<(nbNZ)/static_cast<double>(nbRows*nbCols) ;
    matrix.compress();

    sofa::linearalgebra::FullVector<SReal> known(nbCols), unknown(nbCols), rhs(nbRows);
    for (unsigned i=0; i<nbRows; ++i)
    {
        const auto value = static_cast<SReal>(sofa::helper::drand(1));
        known.set(i,value);
    }
    matrix.mul(rhs, known);

    solver->invert(matrix);
    solver->solve(matrix,  unknown, rhs);

    for (unsigned i=0; i<nbCols; ++i)
    {
        EXPECT_NEAR(unknown[i], known[i], 1e-12);
    }

}
