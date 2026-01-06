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
#include <sofa/component/linearsolver/direct/config.h>

#include <future>
#include <sofa/component/linearsolver/direct/SparseLDLSolver.h>

namespace sofa::component::linearsolver::direct
{

/**
 * This linear solver is based on SparseLDLSolver, a direct linear solver which factorizes the
 * linear system matrix. Its particularity is its asynchronous factorization.
 *
 * The synchronous version performs the following operations (synchronously):
 * 1) Build the matrix
 * 2) Factorize the matrix
 * 3) Solve the system based on the factorization
 *
 * In the asynchronous version, the factorization is performed asynchronously. A consequence is
 * that the solving process uses a factorization which may not be up to date. In practice,
 * the factorization is at least one time step old.
 * Because of this, the solver computes an approximation of the solution, based on an old
 * factorization. It changes the behavior compared to a synchronous version, but it also
 * changes the behavior depending on the duration of the factorization step. It may introduce
 * instabilities.
 */
template<class TMatrix, class TVector, class TThreadManager = NoThreadManager>
class AsyncSparseLDLSolver : public SparseLDLSolver<TMatrix, TVector, TThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(AsyncSparseLDLSolver,TMatrix,TVector,TThreadManager), SOFA_TEMPLATE3(SparseLDLSolver,TMatrix,TVector,TThreadManager));

    using Matrix = typename Inherit1::Matrix;
    using Vector = typename Inherit1::Vector;
    using InvertData = typename Inherit1::InvertData;
    using Real = typename Inherit1::Real;
    using ResMatrixType = typename Inherit1::ResMatrixType;
    using JMatrixType = typename Inherit1::JMatrixType;


    bool isAsyncSolver() override { return true; }

    void init() override;
    void reset() override;

    void solveSystem() override;
    void solve (Matrix& M, Vector& x, Vector& b) override;
    void invert(TMatrix& M) override;
    bool addJMInvJtLocal(TMatrix * M, ResMatrixType * result,const JMatrixType * J, SReal fact) override;

    AsyncSparseLDLSolver();
    ~AsyncSparseLDLSolver() override;

protected:

    /// A second instantiation is needed to differentiate the one that is computed asynchronously and the one that
    /// is used to solve the system in the main thread
    InvertData m_secondInvertData;

    InvertData* m_mainThreadInvertData { nullptr };
    InvertData* m_asyncThreadInvertData { nullptr };

    /// Result of the asynchronous task
    std::future<void> m_asyncResult;

    /// Return true if an asynchronous factorization has been launched and is finished
    bool isAsyncFactorizationFinished() const;

    void launchAsyncFactorization();
    void asyncFactorization();

    bool waitForAsyncTask { true };

    /// Copy the invert data from the async thread to the main thread
    void swapInvertData();

    std::atomic<bool> newInvertDataReady { false };

    Data< bool > d_enableAssembly;
};

#if !defined(SOFA_COMPONENT_LINEARSOLVER_ASYNCSPARSELDLSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API AsyncSparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< SReal>, sofa::linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_DIRECT_API AsyncSparseLDLSolver< sofa::linearalgebra::CompressedRowSparseMatrix< type::Mat<3, 3, SReal> >, sofa::linearalgebra::FullVector<SReal> >;
#endif

} // namespace sofa::component::linearsolver::direct
