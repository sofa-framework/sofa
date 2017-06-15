/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SSORPRECONDITIONER_H
#define SOFA_COMPONENT_LINEARSOLVER_SSORPRECONDITIONER_H
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

/// Linear system solver / preconditioner based on Successive Over Relaxation (SSOR).
///
/// If the matrix is decomposed as $A = D + L + L^T$, this solver computes
//       $(1/(2-w))(D/w+L)(D/w)^{-1}(D/w+L)^T x = b$
//  , or $(D+L)D^{-1}(D+L)^T x = b$ if $w=1$
template<class TMatrix, class TVector, class TThreadManager = NoThreadManager>
class SSORPreconditioner : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE3(SSORPreconditioner,TMatrix,TVector,TThreadManager),SOFA_TEMPLATE3(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector,TThreadManager));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Index Index;
    typedef TThreadManager ThreadManager;
    typedef SReal Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector,TThreadManager> Inherit;

    Data<bool> f_verbose;
    Data<double> f_omega;
protected:
    SSORPreconditioner();
public:
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    MatrixInvertData * createInvertData()
    {
        return new SSORPreconditionerInvertData();
    }

protected :

    class SSORPreconditionerInvertData : public MatrixInvertData
    {
    public :
        unsigned bsize;
        std::vector<double> inv_diag;
    };

};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
