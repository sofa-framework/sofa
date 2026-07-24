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

// This is a private header. Only include from .cpp files that also include
// <Eigen/CholmodSupport>.

#include <sofa/component/linearsolver/direct/EigenSolverFactory.h>
#include <Eigen/CholmodSupport>

namespace sofacholmod
{

/**
 * Supernodal LL^T solver derived directly from CholmodBase (not from
 * CholmodSupernodalLLT) so that the protected m_cholmodFactor and m_cholmod
 * members remain accessible. CholmodSupernodalLLT makes them private via
 * using-declarations, preventing access from further derived classes.
 */
template<typename MatrixType, int UpLo = Eigen::Lower>
class AccessibleCholmodLLT
    : public Eigen::CholmodBase<MatrixType, UpLo, AccessibleCholmodLLT<MatrixType, UpLo>>
{
    using Base = Eigen::CholmodBase<MatrixType, UpLo, AccessibleCholmodLLT>;

public:
    using typename Base::StorageIndex;

    AccessibleCholmodLLT() : Base()
    {
        this->m_cholmod.final_asis = 1;
        this->m_cholmod.supernodal = CHOLMOD_SUPERNODAL;
    }

    cholmod_factor* cholmodFactor() { return this->m_cholmodFactor; }
    cholmod_common& cholmodCommon() { return this->m_cholmod; }
};

/**
 * CHOLMOD-specific proxy that replaces the generic EigenSolverWrapper for
 * CHOLMOD solvers. Exposes the raw CHOLMOD factor and common, enabling
 * optimized partial solves (e.g. for compliance matrix computation).
 */
class CholmodSolverProxy : public sofa::component::linearsolver::direct::BaseEigenSolverProxy
{
    using CholmodEigenMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

    AccessibleCholmodLLT<CholmodEigenMatrix> m_solver;

    // Persistent scratch reused by forwardSolveLP() across calls, so that
    // cholmod_solve2 recycles these buffers instead of allocating/freeing a new
    // dense at every constraint-solve step (matters for many small systems).
    cholmod_dense* m_permutedB = nullptr; // P * B
    cholmod_dense* m_Z         = nullptr; // L^{-1} * P * B (the returned result)
    cholmod_dense* m_Y         = nullptr; // cholmod_solve2 workspace
    cholmod_dense* m_E         = nullptr; // cholmod_solve2 workspace

public:

    ~CholmodSolverProxy() override
    {
        cholmod_common& c = m_solver.cholmodCommon();
        cholmod_free_dense(&m_permutedB, &c);
        cholmod_free_dense(&m_Z, &c);
        cholmod_free_dense(&m_Y, &c);
        cholmod_free_dense(&m_E, &c);
    }

    [[nodiscard]] Eigen::ComputationInfo info() const override
    {
        return m_solver.info();
    }

    void solve(const EigenVectorXdMap<float>& /*b*/, EigenVectorXdMap<float>& /*x*/) const override
    {
        msg_error("CholmodSolverProxy") << "CHOLMOD only supports double precision";
    }

    void solve(const EigenVectorXdMap<double>& b, EigenVectorXdMap<double>& x) const override
    {
        x = m_solver.solve(b);
    }

    void analyzePattern(const BaseEigenSolverProxy::EigenSparseMatrixMap<float>& /*a*/) override
    {
        msg_error("CholmodSolverProxy") << "CHOLMOD only supports double precision";
    }

    void analyzePattern(const BaseEigenSolverProxy::EigenSparseMatrixMap<double>& a) override
    {
        m_solver.analyzePattern(a);
    }

    void factorize(const BaseEigenSolverProxy::EigenSparseMatrixMap<float>& /*a*/) override
    {
        msg_error("CholmodSolverProxy") << "CHOLMOD only supports double precision";
    }

    void factorize(const BaseEigenSolverProxy::EigenSparseMatrixMap<double>& a) override
    {
        m_solver.factorize(a);
    }

    // CHOLMOD-specific accessors for partial solves
    cholmod_factor* cholmodFactor() { return m_solver.cholmodFactor(); }
    cholmod_common& cholmodCommon() { return m_solver.cholmodCommon(); }

    /// Solve Z = L^{-1} * P * B for a dense right-hand side B, using
    /// cholmod_solve2 so the result and workspace buffers are reused across
    /// calls (no per-call allocation). The returned dense is owned by the proxy
    /// and stays valid until the next call to this method or destruction.
    /// Returns nullptr on failure.
    cholmod_dense* forwardSolveLP(cholmod_dense* B)
    {
        cholmod_common& c = m_solver.cholmodCommon();
        cholmod_factor* f = m_solver.cholmodFactor();

        if (!cholmod_solve2(CHOLMOD_P, f, B, nullptr,
                            &m_permutedB, nullptr, &m_Y, &m_E, &c))
        {
            return nullptr;
        }
        if (!cholmod_solve2(CHOLMOD_L, f, m_permutedB, nullptr,
                            &m_Z, nullptr, &m_Y, &m_E, &c))
        {
            return nullptr;
        }
        return m_Z;
    }
};

} // namespace sofacholmod
