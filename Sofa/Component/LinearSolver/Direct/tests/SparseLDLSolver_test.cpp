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
#include <sofa/component/linearsolver/ordering/NaturalOrderingMethod.h>
#include <sofa/component/linearsolver/direct/SparseCommon.h>
#include <sofa/component/linearsystem/MatrixLinearSystem.h>
#include <sofa/simulation/Node.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simpleapi/SimpleApi.h>

#include <sofa/testing/NumericTest.h>
#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>


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

TEST(SparseLDLSolver, Scalar1x1)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // Explicitly set NaturalOrderingMethod to avoid uninitialized permutation
    using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = sofa::core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    MatrixType A(1, 1);
    A.add(0, 0, 2.0);
    A.compress();

    VectorType b(1), x(1);
    b[0] = 4.0;

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    msg_info("Test") << "x[0] = " << x[0];
    std::cout << "[DEBUG_LOG] Scalar1x1: x[0] = " << x[0] << std::endl;
    EXPECT_NEAR(x[0], 2.0, 1e-10) << "Expected 2.0, got " << x[0];
}

TEST(SparseLDLSolver, IdentityMatrix)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // Explicitly set NaturalOrderingMethod
    using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = sofa::core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    constexpr int n = 5;
    MatrixType A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.add(i, i, 1.0);
    }
    A.compress();

    VectorType b(n), x(n);
    for (int i = 0; i < n; ++i)
    {
        b[i] = (SReal)(i + 1);
    }

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(x[i], b[i], 1e-10) << "At index " << i << ", expected " << b[i] << ", got " << x[i];
    }
}

TEST(SparseLDLSolver, BlockMatrix3x3)
{
    using Block = sofa::type::Mat<3, 3, SReal>;
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<Block>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // Explicitly set NaturalOrderingMethod
    using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = sofa::core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    constexpr int nBlocks = 2;
    MatrixType A(nBlocks * 3, nBlocks * 3);

    // [B0 0]
    // [0  B1]
    // where B0 = [1 0 0]
    //            [0 1 0]
    //            [0 0 1]
    // and B1 = [2 0 0]
    //          [0 4 0]
    //          [0 0 8]
    
    // Block 0: Identity
    Block B0; B0.identity();
    A.add(0, 0, B0);
    
    // Block 1: Diagonal
    Block B1; B1.clear();
    B1[0][0] = 2.0; B1[1][1] = 4.0; B1[2][2] = 8.0;
    A.add(3, 3, B1);
    
    A.compress();

    // The system size is nBlocks * 3 = 6
    VectorType b(6), x(6);
    b[0] = 1.0; b[1] = 2.0; b[2] = 3.0;
    b[3] = 4.0; b[4] = 8.0; b[5] = 16.0;

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    // Expected x = [1, 2, 3, 2, 2, 2]
    
    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 2.0, 1e-10);
    EXPECT_NEAR(x[2], 3.0, 1e-10);
    
    EXPECT_NEAR(x[3], 2.0, 1e-10);
    EXPECT_NEAR(x[4], 2.0, 1e-10);
    EXPECT_NEAR(x[5], 2.0, 1e-10);
}

TEST(SparseLDLSolver, BlockMatrix3x3_NonDiagonal)
{
    using Block = sofa::type::Mat<3, 3, SReal>;
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<Block>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // Explicitly set NaturalOrderingMethod
    using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = sofa::core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    constexpr int nBlocks = 1;
    MatrixType A(nBlocks * 3, nBlocks * 3);

    // Symmetric 3x3 block
    // [ 2  1  0 ]
    // [ 1  2  1 ]
    // [ 0  1  2 ]
    Block B;
    B.clear();
    B[0][0] = 2.0; B[0][1] = 1.0; B[0][2] = 0.0;
    B[1][0] = 1.0; B[1][1] = 2.0; B[1][2] = 1.0;
    B[2][0] = 0.0; B[2][1] = 1.0; B[2][2] = 2.0;
    A.addBlock(0, 0, B);

    A.compress();

    VectorType b(3), x(3);
    // b = A * [1, 1, 1] = [3, 4, 3]
    b[0] = 3.0; b[1] = 4.0; b[2] = 3.0;

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
}

TEST(SparseLDLSolver, DiagonalMatrix)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    const int n = 5;
    MatrixType A(n, n);
    for (int i = 0; i < n; ++i)
        A.add(i, i, (SReal)(i + 1));
    A.compress();

    VectorType b(n), x(n);
    for (int i = 0; i < n; ++i)
        b[i] = 1.0;

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    for (int i = 0; i < n; ++i)
        EXPECT_NEAR(x[i], 1.0 / (i + 1), 1e-10);
}

TEST(SparseLDLSolver, SimpleSPD2x2)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // A = [ 2 -1 ]
    //     [ -1 2 ]
    MatrixType A(2, 2);
    A.add(0, 0, 2.0);
    A.add(0, 1, -1.0);
    A.add(1, 0, -1.0);
    A.add(1, 1, 2.0);
    A.compress();

    VectorType b(2), x(2);
    b[0] = 1.0;
    b[1] = 0.0;
    // Expected x = [2/3, 1/3]

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    EXPECT_NEAR(x[0], 2.0/3.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0/3.0, 1e-10);
}

TEST(SparseLDLSolver, SimpleSPD3x3)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    // A = [ 4  12 -16 ]
    //     [ 12 37 -43 ]
    //     [-16 -43 98 ]
    MatrixType A(3, 3);
    A.add(0, 0, 4.0); A.add(0, 1, 12.0); A.add(0, 2, -16.0);
    A.add(1, 0, 12.0); A.add(1, 1, 37.0); A.add(1, 2, -43.0);
    A.add(2, 0, -16.0); A.add(2, 1, -43.0); A.add(2, 2, 98.0);
    A.compress();

    VectorType b(3), x(3);
    b[0] = 1.0; b[1] = 2.0; b[2] = 3.0;

    solver->init();
    solver->invert(A);
    solver->solve(A, x, b);

    // Verify A*x = b
    VectorType Ax(3);
    for(int i=0; i<3; ++i) {
        Ax[i] = 0;
        for(int j=0; j<3; ++j) Ax[i] += A(i,j) * x[j];
    }

    EXPECT_NEAR(Ax[0], b[0], 1e-10);
    EXPECT_NEAR(Ax[1], b[1], 1e-10);
    EXPECT_NEAR(Ax[2], b[2], 1e-10);
}

TEST(SparseLDLSolver, MultiStepFactorization)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = sofa::component::linearsolver::direct::SparseLDLSolver<MatrixType, VectorType>;
    const Solver::SPtr solver = sofa::core::objectmodel::New<Solver>();

    const int n = 3;
    MatrixType A(n, n);
    A.add(0, 0, 2.0); A.add(0, 1, -1.0);
    A.add(1, 0, -1.0); A.add(1, 1, 2.0); A.add(1, 2, -1.0);
    A.add(2, 1, -1.0); A.add(2, 2, 2.0);
    A.compress();

    VectorType b(n), x(n);
    for(int i=0; i<n; ++i) b[i] = 1.0;

    solver->init();
    
    // First solve
    solver->invert(A);
    solver->solve(A, x, b);
    
    // Change values but NOT shape
    {
        auto& values = const_cast<MatrixType::VecBlock&>(A.getColsValue());
        values[0] = 4.0; // A(0,0)
    }
    solver->invert(A);
    solver->solve(A, x, b);

    VectorType Ax(n);
    for(int i=0; i<n; ++i)
    {
        Ax[i] = 0;
        for(int j=0; j<n; ++j)
        {
            Ax[i] += A(i,j) * x[j];
        }
    }
    for(int i=0; i<n; ++i)
    {
        EXPECT_NEAR(Ax[i], b[i], 1e-10);
    }
}
