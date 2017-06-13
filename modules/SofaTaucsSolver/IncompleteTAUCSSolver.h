/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_IncompleteTAUCSSolver_H
#define SOFA_COMPONENT_LINEARSOLVER_IncompleteTAUCSSolver_H
#include "config.h"

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/MechanicalVisitor.h>
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
    Data<int>    f_incompleteType; ///< 0 = Incomplete Cholesky (ic), 1 = vaidya (va) , 2 = recursive vaidya (vr) 
#endif
    Data<int>    f_ordering; ///< ose ordering 0=identity/1=tree/2=metis/3=natural/4=genmmd/5=md/6=mmd/7=amd
    Data<double> f_dropTol; ///< Drop tolerance use for incomplete factorization
    Data<bool>   f_modified_flag; ///< Modified ICC : maintains rowsums
#ifdef VAIDYA
    Data<double> f_subgraphs; ///< Desired number of subgraphs for Vaidya
    Data<bool>   f_stretch_flag; ///< Starting with a low stretch tree
    Data<bool>   f_multifrontal; ///< Use multifrontal algo in vaidya
    Data<int>    f_seed; ///< Affects decomposition to subtrees
    Data<double> f_C; ///< splits tree into about k=f_C*n^(1/(1+f_epsilon)) subgraphs
    Data<double> f_epsilon; ///< splits tree into about k=f_C*n^(1/(1+f_epsilon)) subgraphs
    Data<int>    f_nsmall; ///< matrices smaller than nsmall are factored directly
    Data<int>    f_maxlevels; ///< preconditioner has at most l levels
    Data<int>    f_innerits; ///< using at most m iterations
    Data<double> f_innerconv; ///< inner solves reduce their residual by a factor of r
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
