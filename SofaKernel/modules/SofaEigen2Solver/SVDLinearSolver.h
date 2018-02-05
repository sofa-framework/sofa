/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SVDLinearSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_SVDLinearSOLVER_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/helper/map.h>
#include <math.h>

namespace sofa
{
namespace component
{
namespace linearsolver
{

/** Linear system solver using the JacobiSVD decomposition of the Eigen library (http://eigen.tuxfamily.org/  , see also an excellent introduction in Numerical Recipes.)
  The Singular Value Decomposition is probably the most robust (and the slowest !) matrix factoring for linear equation solution.
  It works only for dense matrices (FullMatrix).
  The equation system Ax=b is solved using a decomposition of A=USV^T, where U is a n-by-n unitary, V is a p-by-p unitary, and S is a n-by-p real positive matrix which is zero outside of its main diagonal; the diagonal entries of S are known as the singular values of A and the columns of U and V are known as the left and right singular vectors of A respectively.
  In case of indefinite matrix, there is at least one null singular value, and there is no solution to the equation system except for special right-hand terms.
  The SVD solver solves the equation in the least-square sense: it finds the pseudo-solution x which minimizes Ax-b.
  The condition number of the matrix is a byproduct of the solution, written in attribute "conditionNumber" by method solve (Matrix& M, Vector& x, Vector& b).
  */

template<class TMatrix, class TVector>
class SOFA_EIGEN2_SOLVER_API SVDLinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SVDLinearSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename TVector::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<bool> f_verbose;
    Data<Real> f_minSingularValue;
protected:
    SVDLinearSolver();
public:
    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b) override;
    Data<Real> f_conditionNumber; ///< Condition number of the matrix: ratio between the largest and smallest singular values. Computed in method solve.

#ifdef DISPLAY_TIME
    double timeStamp;
#endif
};

} // namespace linearsolver
} // namespace component
} // namespace sofa

#endif
