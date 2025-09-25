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
#include <sofa/component/linearsolver/iterative/config.h>

#include <sofa/component/linearsolver/iterative/MatrixLinearSolver.h>
#include <sofa/helper/map.h>

namespace sofa::component::linearsolver::iterative
{

/// Linear system solver using the conjugate gradient iterative algorithm
template<class TMatrix, class TVector>
class CGLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix, TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CGLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;
    using Real = typename Matrix::Real;

    Data<unsigned> d_maxIter; ///< Maximum number of iterations after which the iterative descent of the Conjugate Gradient must stop
    Data<Real> d_tolerance; ///< Desired accuracy of the Conjugate Gradient solution evaluating: |r|²/|b|² (ratio of current residual norm over initial residual norm)
    Data<Real> d_smallDenominatorThreshold; ///< Minimum value of the denominator (pT A p)^ in the conjugate Gradient solution
    Data<bool> d_warmStart; ///< Use previous solution as initial solution, which may improve the initial guess if your system is evolving smoothly
    Data<std::map < std::string, sofa::type::vector<Real> > > d_graph; ///< Graph of residuals at each iteration

protected:

    CGLinearSolver();

    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: p = p*beta + r
    inline void cgstep_beta(const core::ExecParams* params, Vector& p, Vector& r, Real beta);
    /// This method is separated from the rest to be able to use custom/optimized versions depending on the types of vectors.
    /// It computes: x += p*alpha, r -= q*alpha
    inline void cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, Real alpha);

    int timeStepCount{0};
    bool equilibriumReached{false};

public:
    void init() override;
    void reinit() override {};

    /// Solve iteratively the linear system Ax=b following a conjugate gradient descent
    void solve (Matrix& A, Vector& x, Vector& b) override;
};

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_beta(const core::ExecParams* /*params*/, Vector& p, Vector& r, Real beta);

template<>
inline void CGLinearSolver<component::linearsolver::GraphScatteredMatrix,component::linearsolver::GraphScatteredVector>::cgstep_alpha(const core::ExecParams* params, Vector& x, Vector& r, Vector& p, Vector& q, Real alpha);

#if !defined(SOFA_COMPONENT_LINEARSOLVER_CGLINEARSOLVER_CPP)
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< GraphScatteredMatrix, GraphScatteredVector >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::FullMatrix<SReal>, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::SparseMatrix<SReal>, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<SReal>, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, linearalgebra::FullVector<SReal> >;
extern template class SOFA_COMPONENT_LINEARSOLVER_ITERATIVE_API CGLinearSolver< linearalgebra::CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, linearalgebra::FullVector<SReal> >;


#endif

} // namespace sofa::component::linearsolver::iterative
