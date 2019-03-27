/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_LULINEARSOLVER_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <cmath>

namespace sofa
{

namespace component
{

namespace linearsolver
{

/// Linear system solver using the default (LU factorization) algorithm
template<class Matrix, class Vector>
class LULinearSolver : public sofa::component::linearsolver::MatrixLinearSolver<Matrix,Vector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(LULinearSolver,Matrix,Vector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,Matrix,Vector));

    Data<bool> f_verbose; ///< Dump system state at each iteration
    typename Matrix::LUSolver* solver;
    typename Matrix::InvMatrixType Minv;
    bool computedMinv;
protected:
    LULinearSolver();
    ~LULinearSolver();

public:
    /// Invert M
    void invert (Matrix& M) override;

    /// Solve Mx=b
    void solve (Matrix& M, Vector& x, Vector& b) override;

    void computeMinv();
    double getMinvElement(int i, int j);

    template<class RMatrix, class JMatrix>
    bool addJMInvJt(RMatrix& result, JMatrix& J, double fact);

    /// Multiply the inverse of the system matrix by the transpose of the given matrix, and multiply the result with the given matrix J
    ///
    /// @param result the variable where the result will be added
    /// @param J the matrix J to use
    /// @return false if the solver does not support this operation, of it the system matrix is not invertible
    bool addJMInvJt(defaulttype::BaseMatrix* result, defaulttype::BaseMatrix* J, double fact) override;

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
