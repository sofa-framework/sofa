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
#ifndef SOFA_COMPONENT_LINEARSOLVER_IncompleteTAUCSSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_IncompleteTAUCSSolver_H

#include <sofa/core/behavior/LinearSolver.h>
#include <sofa/component/linearsolver/ParallelMatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>

// include all headers included in taucs.h to fix errors on macx
#ifndef WIN32
#include <complex.h>
#endif

#include <assert.h>
#include <float.h>
#include <stdlib.h>

extern "C" {
#include <taucs.h>
}

namespace sofa
{

namespace component
{

namespace linearsolver
{



/// Direct linear solvers implemented with the TAUCS library
template<class TMatrix, class TVector>
class IncompleteTAUCSSolver : public sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(IncompleteTAUCSSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::ParallelMatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::ParallelMatrixLinearSolver<TMatrix,TVector> Inherit;
    typedef sofa::core::behavior::BaseMechanicalState::VecId VecId;

    Data<int> f_incompleteType;
    Data<int> f_ordering;
    Data<double> f_dropTol;
    Data<bool> f_modified_flag;
    Data<double> f_subgraphs;
    Data<bool> f_stretch_flag;
    Data<bool> f_multifrontal;
    Data<int> f_seed;

    IncompleteTAUCSSolver();
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

protected:
    class IncompleteTAUCSSolverInvertData : public defaulttype::MatrixInvertData
    {
    public :
        Matrix Mfiltered;
        int* perm;
        int* invperm;
        taucs_ccs_matrix matrix_taucs;
        taucs_ccs_matrix* L;
        helper::vector<double> B;
        helper::vector<double> R;

        IncompleteTAUCSSolverInvertData()
        {
            perm    = NULL;
            invperm = NULL;
            L       = NULL;
        }

        ~IncompleteTAUCSSolverInvertData()
        {
            if (perm) taucs_free(perm);
            if (invperm) taucs_free(invperm);
            if (L) taucs_ccs_free(L);
            perm    = NULL;
            invperm = NULL;
            L       = NULL;
        }
    };

    bool new_perm;
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
