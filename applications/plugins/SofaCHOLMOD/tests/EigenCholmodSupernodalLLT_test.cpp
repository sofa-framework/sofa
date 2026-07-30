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

#include <SofaCHOLMOD/EigenCholmodSupernodalLLT.h>
#include <SofaCHOLMOD/CholmodSolverProxy.h>
#include <SofaCHOLMOD/init.h>
#include <sofa/component/linearsolver/ordering/NaturalOrderingMethod.h>

#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/type/Mat.h>

#include <Eigen/Dense>

namespace sofacholmod
{

using sofa::core::objectmodel::New;
using NaturalOrdering = sofa::component::linearsolver::ordering::NaturalOrderingMethod;

namespace
{
/// Instantiate a CHOLMOD solver with a NaturalOrderingMethod and init() it.
/// sofacholmod::init() populates the internal solver factory (idempotent).
template<class Solver>
typename Solver::SPtr makeSolver(NaturalOrdering::SPtr& orderingOut)
{
    sofacholmod::init();
    auto solver = New<Solver>();
    orderingOut = New<NaturalOrdering>();
    solver->l_orderingMethod.set(orderingOut.get());
    solver->init();
    return solver;
}
}

/// The factory must produce a CholmodSolverProxy (not the generic wrapper),
/// otherwise the optimized addJMInvJtLocal path is silently bypassed and the
/// solver falls back to a slow full-solve-per-constraint-row implementation.
TEST(EigenCholmodSupernodalLLT, FactoryRegistersCholmodProxy)
{
    sofacholmod::init();

    ASSERT_TRUE(MainCholmodSupernodalLLTFactory::hasSolver<SReal>("Natural"));

    auto* proxy = MainCholmodSupernodalLLTFactory::getSolver<SReal>("Natural");
    ASSERT_NE(proxy, nullptr);
    EXPECT_NE(dynamic_cast<CholmodSolverProxy*>(proxy), nullptr)
        << "the CHOLMOD factory must produce a CholmodSolverProxy";
    delete proxy;
}

/// Solve a SPD tridiagonal system and check the residual A*x == b.
TEST(EigenCholmodSupernodalLLT, SolveSPDScalar)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = EigenCholmodSupernodalLLT<SReal>;

    NaturalOrdering::SPtr ordering;
    const Solver::SPtr solver = makeSolver<Solver>(ordering);

    constexpr int n = 6;
    MatrixType A(n, n);
    for (int i = 0; i < n; ++i)
    {
        A.add(i, i, 2.0 + 0.1 * i); // SPD, diagonally dominant
        if (i > 0)
        {
            A.add(i, i - 1, -1.0);
            A.add(i - 1, i, -1.0);
        }
    }
    A.compress();

    VectorType b(n), x(n);
    for (int i = 0; i < n; ++i)
    {
        b[i] = static_cast<SReal>(i + 1);
    }

    solver->invert(A);
    solver->solve(A, x, b);

    for (int i = 0; i < n; ++i)
    {
        SReal Ax = 0;
        for (int j = 0; j < n; ++j)
        {
            Ax += A(i, j) * x[j];
        }
        EXPECT_NEAR(Ax, b[i], 1e-9) << "residual mismatch at row " << i;
    }
}

/// Same, using the Mat3x3 block template.
TEST(EigenCholmodSupernodalLLT, SolveSPDBlock3x3)
{
    using Block = sofa::type::Mat<3, 3, SReal>;
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<Block>;
    using VectorType = sofa::linearalgebra::FullVector<SReal>;
    using Solver = EigenCholmodSupernodalLLT<Block>;

    NaturalOrdering::SPtr ordering;
    const Solver::SPtr solver = makeSolver<Solver>(ordering);

    // Two SPD diagonal blocks.
    constexpr int n = 6;
    MatrixType A(n, n);
    Block B0; B0.identity();
    Block B1; B1.clear(); B1[0][0] = 2.0; B1[1][1] = 4.0; B1[2][2] = 8.0;
    A.add(0, 0, B0);
    A.add(3, 3, B1);
    A.compress();

    VectorType b(n), x(n);
    b[0] = 1.0; b[1] = 2.0; b[2] = 3.0;
    b[3] = 4.0; b[4] = 8.0; b[5] = 16.0;

    solver->invert(A);
    solver->solve(A, x, b);

    const SReal expected[n] = {1.0, 2.0, 3.0, 2.0, 2.0, 2.0};
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(x[i], expected[i], 1e-9) << "at index " << i;
    }
}

/// Verify the CHOLMOD-specific addJMInvJtLocal against a dense reference
/// W = fact * J * A^{-1} * J^T. This exercises the CholmodSolverProxy path
/// (forward-solve via cholmod_solve2 + the dsyrk symmetric product).
TEST(EigenCholmodSupernodalLLT, AddJMInvJtMatchesDenseReference)
{
    using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
    using Solver = EigenCholmodSupernodalLLT<SReal>;

    NaturalOrdering::SPtr ordering;
    const Solver::SPtr solver = makeSolver<Solver>(ordering);

    constexpr int n = 6;
    MatrixType A(n, n);
    Eigen::MatrixXd Adense = Eigen::MatrixXd::Zero(n, n);
    const auto addA = [&](int i, int j, double v) { A.add(i, j, v); Adense(i, j) += v; };
    for (int i = 0; i < n; ++i)
    {
        addA(i, i, 2.0 + 0.1 * i);
        if (i > 0)
        {
            addA(i, i - 1, -1.0);
            addA(i - 1, i, -1.0);
        }
    }
    A.compress();

    solver->invert(A);
    // A directly-instantiated solver is not necessarily left in the Valid state
    // by init(); addJMInvJtLocal early-returns unless the component is valid.
    solver->d_componentState.setValue(sofa::core::objectmodel::ComponentState::Valid);

    // Constraint Jacobian J (m x n), sparse.
    constexpr int m = 3;
    Solver::JMatrixType J;
    J.resize(m, n);
    Eigen::MatrixXd Jdense = Eigen::MatrixXd::Zero(m, n);
    const auto setJ = [&](int i, int j, double v) { J.set(i, j, v); Jdense(i, j) = v; };
    setJ(0, 0, 1.0); setJ(0, 2, -0.5);
    setJ(1, 1, 2.0); setJ(1, 4, 1.0);
    setJ(2, 3, 1.0); setJ(2, 5, -1.0);

    const SReal fact = 0.75;

    sofa::linearalgebra::FullMatrix<SReal> W;
    W.resize(m, m);

    const bool ok = solver->addJMInvJtLocal(&A, &W, &J, fact);
    ASSERT_TRUE(ok);

    const Eigen::MatrixXd Wref = fact * Jdense * Adense.inverse() * Jdense.transpose();

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            EXPECT_NEAR(W.element(i, j), Wref(i, j), 1e-8)
                << "compliance mismatch at (" << i << "," << j << ")";
        }
    }
}

} // namespace sofacholmod
