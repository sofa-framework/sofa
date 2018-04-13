/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_CholeskySolver_H
#define SOFA_COMPONENT_LINEARSOLVER_CholeskySolver_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <sofa/helper/map.h>

#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Direct linear solver based on Cholesky factorization, for dense matrices
template<class TMatrix, class TVector>
class CholeskySolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(CholeskySolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Vector::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<bool> f_verbose; ///< Dump system state at each iteration

    CholeskySolver();

    /// Compute x such as Mx=b. M is not used, it must have been factored before using method invert(Matrix& M)
    void solve (Matrix& M, Vector& x, Vector& b) override;

    /// Factors the matrix. Must be done before solving
    void invert(Matrix& M) override;

private :
    FullMatrix<typename Vector::Real> L;
};

#if defined(SOFA_EXTERN_TEMPLATE) && !defined(SOFA_COMPONENT_LINEARSOLVER_CHOLESKYSOLVER_CPP)
#ifndef SOFA_FLOAT
extern template class SOFA_GENERAL_LINEAR_SOLVER_API CholeskySolver< SparseMatrix<double>, FullVector<double> >;
extern template class SOFA_GENERAL_LINEAR_SOLVER_API CholeskySolver< FullMatrix<double>, FullVector<double> >;
#endif
#ifndef SOFA_DOUBLE
extern template class SOFA_GENERAL_LINEAR_SOLVER_API CholeskySolver< SparseMatrix<float>, FullVector<float> >;
extern template class SOFA_GENERAL_LINEAR_SOLVER_API CholeskySolver< FullMatrix<float>, FullVector<float> >;
#endif
#endif

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
