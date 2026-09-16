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

#include <sofa/component/linearsolver/direct/EigenSimplicialLDLT.h>
#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.inl>
#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>
#include <sofa/component/linearsolver/ordering/NaturalOrderingMethod.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/FullVector.h>

namespace sofa
{

using MatrixType = sofa::linearalgebra::CompressedRowSparseMatrix<SReal>;
using VectorType = sofa::linearalgebra::FullVector<SReal>;

/// Gives access to the filtered matrix, i.e. the one which is handed to Eigen
class TestableEigenSimplicialLDLT : public component::linearsolver::direct::EigenSimplicialLDLT<SReal>
{
public:
    SOFA_CLASS(TestableEigenSimplicialLDLT, SOFA_TEMPLATE(component::linearsolver::direct::EigenSimplicialLDLT, SReal));

    const MatrixType& filteredMatrix() const { return this->Mfiltered; }
};

/// Builds a matrix in which the last row and column are entirely zero
static MatrixType matrixWithAnEmptyRow()
{
    MatrixType A(3, 3);
    A.add(0, 0, 2.0);
    A.add(1, 1, 3.0);
    // nothing is added on row and column 2
    A.compress();
    return A;
}

static TestableEigenSimplicialLDLT::SPtr createSolver()
{
    // the ordering methods are registered when the module is initialized, which does not
    // happen here since the test links the library directly
    component::linearsolver::direct::MainSimplicialLDLTFactory::registerSolver<
        Eigen::NaturalOrdering<int>, SReal>("Natural");

    const TestableEigenSimplicialLDLT::SPtr solver =
        core::objectmodel::New<TestableEigenSimplicialLDLT>();

    using NaturalOrdering = component::linearsolver::ordering::NaturalOrderingMethod;
    const NaturalOrdering::SPtr ordering = core::objectmodel::New<NaturalOrdering>();
    solver->l_orderingMethod.set(ordering.get());

    solver->init();
    return solver;
}

/// The filtered matrix is mapped by Eigen as it is: it reads one row offset per row, plus
/// one. Since copyNonZeros() does not store the rows which are entirely zero, they must be
/// restored, otherwise the mapping reads past the end of the arrays.
TEST(EigenDirectSparseSolver, filteredMatrixHasOneOffsetPerRow)
{
    const auto solver = createSolver();

    MatrixType A = matrixWithAnEmptyRow();
    solver->invert(A);

    const MatrixType& filtered = solver->filteredMatrix();
    EXPECT_EQ(filtered.rows(), 3);
    EXPECT_EQ(filtered.rowBegin.size(), static_cast<std::size_t>(filtered.rows()) + 1);
}

/// The rows are restored at every call, so a matrix keeping the same pattern over several
/// steps stays consistent.
TEST(EigenDirectSparseSolver, filteredMatrixHasOneOffsetPerRowOverSeveralSteps)
{
    const auto solver = createSolver();

    MatrixType A = matrixWithAnEmptyRow();

    for (unsigned int step = 0; step < 3; ++step)
    {
        solver->invert(A);

        const MatrixType& filtered = solver->filteredMatrix();
        EXPECT_EQ(filtered.rowBegin.size(), static_cast<std::size_t>(filtered.rows()) + 1)
            << "at step " << step;
    }
}

/// A matrix with an empty row is singular: the solver must report it instead of failing in
/// an uncontrolled way. Under a sanitizer, this also covers the mapping itself.
TEST(EigenDirectSparseSolver, factorizeMatrixWithAnEmptyRow)
{
    // required to be able to use EXPECT_MSG_EMIT
    helper::logging::MessageDispatcher::addHandler(testing::MainGtestMessageHandler::getInstance());

    const auto solver = createSolver();

    MatrixType A = matrixWithAnEmptyRow();

    {
        EXPECT_MSG_EMIT(Error);
        solver->invert(A);
    }

    VectorType b(3), x(3);
    b[0] = 1.0; b[1] = 1.0; b[2] = 1.0;
    solver->solve(A, x, b);
}

/// CompressedRowSparseMatrix accepts entries outside its declared size and grows silently
/// to store them, which leaves more row offsets than rows. Such a matrix cannot be mapped
/// by Eigen: the solver reports it instead of reading past the end of the arrays.
TEST(EigenDirectSparseSolver, factorizeMatrixWithEntriesOutOfBounds)
{
    helper::logging::MessageDispatcher::addHandler(testing::MainGtestMessageHandler::getInstance());

    const auto solver = createSolver();

    MatrixType A(3, 3);
    A.add(0, 0, 2.0);
    A.add(1, 1, 3.0);
    A.add(2, 2, 4.0);
    A.add(5, 5, 1.0); // outside the declared size of the matrix
    A.compress();

    {
        EXPECT_MSG_EMIT(Error);
        solver->invert(A);
    }

    // the solver is left unusable rather than solving with a corrupted matrix
    EXPECT_TRUE(solver->isComponentStateInvalid());

    VectorType b(3), x(3);
    b[0] = 1.0; b[1] = 1.0; b[2] = 1.0;
    solver->solve(A, x, b);
}

/// A matrix without any empty row is not affected: it is factorized and solved as usual.
TEST(EigenDirectSparseSolver, factorizeMatrixWithoutEmptyRow)
{
    helper::logging::MessageDispatcher::addHandler(testing::MainGtestMessageHandler::getInstance());

    const auto solver = createSolver();

    MatrixType A(3, 3);
    A.add(0, 0, 2.0);
    A.add(1, 1, 3.0);
    A.add(2, 2, 4.0);
    A.compress();

    VectorType b(3), x(3);
    b[0] = 2.0; b[1] = 3.0; b[2] = 4.0;

    {
        EXPECT_MSG_NOEMIT(Error);
        solver->invert(A);
        solver->solve(A, x, b);
    }

    const MatrixType& filtered = solver->filteredMatrix();
    EXPECT_EQ(filtered.rowBegin.size(), static_cast<std::size_t>(filtered.rows()) + 1);

    EXPECT_NEAR(x[0], 1.0, 1e-10);
    EXPECT_NEAR(x[1], 1.0, 1e-10);
    EXPECT_NEAR(x[2], 1.0, 1e-10);
}

}
