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
#include <SofaCHOLMOD/config.h>

#include <sofa/component/linearsolver/direct/EigenDirectSparseSolver.h>
#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>
#include <sofa/core/objectmodel/Data.h>

namespace sofacholmod
{

/**
 * Singleton factory dedicated to CHOLMOD supernodal LLT solvers.
 *
 * CHOLMOD manages its own fill-reducing ordering internally (AMD, METIS, NESDIS),
 * so the OrderingMethodType template parameter is intentionally ignored.
 * The ordering method name is only used as a key for the factory lookup.
 *
 * Unlike other factories, registerSolver is not defined in this header because
 * it requires <Eigen/CholmodSupport> which we don't want to pull into every
 * translation unit. It is defined in init.cpp instead.
 */
class SOFACHOLMOD_API MainCholmodSupernodalLLTFactory : public sofa::component::linearsolver::direct::BaseMainEigenSolverFactory<MainCholmodSupernodalLLTFactory>
{
   public:
    ~MainCholmodSupernodalLLTFactory();

    template<typename OrderingMethodType, class ScalarType>
    static void registerSolver(const std::string& orderingMethodName);
};


/**
 * Direct linear solver based on a supernodal sparse LL^T Cholesky factorization
 * using CHOLMOD from the SuiteSparse library (via Eigen's CholmodSupport).
 *
 * This solver is significantly faster than simplicial solvers (SparseLDLSolver,
 * EigenSimplicialLDLT) for medium-to-large systems (n > ~2000) thanks to
 * supernodal BLAS3-based factorization. CHOLMOD automatically selects the best
 * fill-reducing ordering (AMD, METIS, or NESDIS) regardless of the linked
 * OrderingMethod component.
 */
template<class TBlockType>
class EigenCholmodSupernodalLLT
    : public sofa::component::linearsolver::direct::EigenDirectSparseSolver<
        TBlockType,
        MainCholmodSupernodalLLTFactory
    >
{
public:
    typedef sofa::linearalgebra::CompressedRowSparseMatrix<TBlockType> Matrix;
    using Real = typename Matrix::Real;
    typedef sofa::linearalgebra::FullVector<Real> Vector;

    SOFA_CLASS(SOFA_TEMPLATE(EigenCholmodSupernodalLLT, TBlockType), SOFA_TEMPLATE2(sofa::component::linearsolver::direct::EigenDirectSparseSolver, TBlockType, MainCholmodSupernodalLLTFactory));

    using JMatrixType = typename Inherit1::JMatrixType;
    using ResMatrixType = typename Inherit1::ResMatrixType;

    void init() override;
    void reinit() override;
    void invert(Matrix& A) override;

    bool addJMInvJtLocal(Matrix* M, ResMatrixType* result,
                         const JMatrixType* J, SReal fact) override;

    /// Number of threads the underlying BLAS (used by CHOLMOD's supernodal
    /// factorization) is allowed to use.
    /// Default is 1, which is the fastest setting for the vast majority of SOFA
    /// scenes: it avoids thread oversubscription, which is catastrophic when
    /// several solvers are factorized in parallel (e.g. with parallelODESolving)
    /// and still optimal for a single medium-sized system. Increase it only for
    /// a single, very large standalone system. A value <= 0 leaves the BLAS
    /// default untouched, i.e. controlled by the OPENBLAS_NUM_THREADS /
    /// OMP_NUM_THREADS environment variables. Only effective with OpenBLAS or
    /// MKL; ignored with BLAS backends that expose no runtime thread-control API
    /// (e.g. Apple Accelerate), where the environment variables must be used
    /// instead.
    sofa::core::objectmodel::Data<int> d_numThreads;

protected:
    EigenCholmodSupernodalLLT();

    /// Apply d_numThreads to the underlying BLAS backend if it exposes a runtime
    /// thread-count API (OpenBLAS/MKL). No-op if d_numThreads <= 0.
    void applyBlasNumThreads();

    /// Reused dense J^T buffer for addJMInvJtLocal, to avoid reallocating at every
    /// constraint-solve step. Not thread-safe, but each solver instance handles a
    /// single object whose compliance is built sequentially.
    Eigen::MatrixXd m_Jt;
};

#ifndef SOFACHOLMOD_EIGENCHOLMODSUPERNODALLLT_CPP
extern template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< SReal >;
extern template class SOFACHOLMOD_API EigenCholmodSupernodalLLT< sofa::type::Mat<3,3,SReal> >;
#endif

} // namespace sofacholmod
