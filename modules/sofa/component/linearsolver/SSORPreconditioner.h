/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef SOFA_COMPONENT_LINEARSOLVER_SSORPRECONDITIONER_H
#define SOFA_COMPONENT_LINEARSOLVER_SSORPRECONDITIONER_H

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <sofa/component/linearsolver/ParallelMatrixLinearSolver.inl>

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



template<class TMatrix, class TVector>
class SSORPreconditioner : public sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SSORPreconditioner,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::ParallelMatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef SReal Real;
    typedef sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<bool> f_verbose;
    Data<double> f_omega;

    SSORPreconditioner();
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
