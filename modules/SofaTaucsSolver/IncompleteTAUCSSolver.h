/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>

// include all headers included in taucs.h to fix errors on macx
#ifndef WIN32
#include <complex.h>
#endif

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include <taucs_lib.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

//#define VAIDYA

/// Direct linear solvers implemented with the TAUCS library
template<class TMatrix, class TVector>
class IncompleteTAUCSSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(IncompleteTAUCSSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

#ifdef VAIDYA
    Data<int>    f_incompleteType;
#endif
    Data<int>    f_ordering;
    Data<double> f_dropTol;
    Data<bool>   f_modified_flag;
#ifdef VAIDYA
    Data<double> f_subgraphs;
    Data<bool>   f_stretch_flag;
    Data<bool>   f_multifrontal;
    Data<int>    f_seed;
    Data<double> f_C;
    Data<double> f_epsilon;
    Data<int>    f_nsmall;
    Data<int>    f_maxlevels;
    Data<int>    f_innerits;
    Data<double> f_innerconv;
#endif

    IncompleteTAUCSSolver();
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    MatrixInvertData * createInvertData()
    {
        return new IncompleteTAUCSSolverInvertData();
    }

protected:
    typedef struct
    {
        taucs_ccs_matrix** B;
        taucs_ccs_matrix** S;
        taucs_ccs_matrix** L;
        int             levels;
        int             level;
        double          convratio;
        double          maxits;
    } recvaidya_args;

    class IncompleteTAUCSSolverInvertData : public MatrixInvertData
    {
    public :
        CompressedRowSparseMatrix<double> Mfiltered;
        int* perm;
        int* invperm;
        taucs_ccs_matrix matrix_taucs;
        void*            precond_args;
        helper::vector<double> B;
        helper::vector<double> R;
        int             (*precond_fn)(void*,void* x,void* b);
        int n;
        taucs_ccs_matrix * L;
        recvaidya_args * RL;

        IncompleteTAUCSSolverInvertData()
        {
            perm         = NULL;
            invperm      = NULL;
            precond_args = NULL;
            RL           = NULL;
            L            = NULL;
            n = 0;
        }

        ~IncompleteTAUCSSolverInvertData()
        {
            if (perm) taucs_free(perm);
            if (invperm) taucs_free(invperm);
            freeL();
            freeRL();
            perm         = NULL;
            invperm      = NULL;
            precond_args = NULL;
            n = 0;
        }

        void freeL()
        {
            if (L) taucs_ccs_free(L);
            L = NULL;
        }

        void freeRL()
        {
            if (RL)
            {
                for (int i=0; i<32; i++) if (RL->S[i]) taucs_ccs_free(RL->S[i]);
                free(RL->S);
                for (int i=0; i<32; i++) if (RL->L[i]) taucs_ccs_free(RL->L[i]);
                free(RL->L);
            }
            RL = NULL;
        }
    };
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
