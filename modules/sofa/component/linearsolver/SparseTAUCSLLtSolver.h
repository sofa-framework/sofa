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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SPARSETAUCSLLTSOLVER_H
#define SOFA_COMPONENT_LINEARSOLVER_SPARSETAUCSLLTSOLVER_H

#include <sofa/core/behavior/LinearSolver.h>
#include <SofaBaseLinearSolver/MatrixLinearSolver.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/helper/map.h>
#include <math.h>

// include all headers included in taucs.h to fix errors on macx
#ifndef WIN32
#include <complex.h>
#endif

#include <assert.h>
#include <float.h>
#include <stdlib.h>

#include <taucs-mt.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{



/// Direct linear solvers implemented with the TAUCS library
template<class TMatrix, class TVector>
class SOFA_TAUCS_SOLVER_API SparseTAUCSLLtSolver : public sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector>
{
public:
    SOFA_CLASS(SOFA_TEMPLATE2(SparseTAUCSLLtSolver,TMatrix,TVector),SOFA_TEMPLATE2(sofa::component::linearsolver::MatrixLinearSolver,TMatrix,TVector));

    typedef TMatrix Matrix;
    typedef TVector Vector;
    typedef typename Matrix::Real Real;
    typedef sofa::component::linearsolver::MatrixLinearSolver<TMatrix,TVector> Inherit;

    Data<bool> f_verbose;
    Data<double> f_dropTol;
    Data<unsigned> f_nproc_simu;

    SparseTAUCSLLtSolver();
    void solve (Matrix& M, Vector& x, Vector& b);
    void invert(Matrix& M);

    MatrixInvertData * createInvertData()
    {
        return new SparseTAUCSLLtSolverInvertData();
    }

protected:
    class SparseTAUCSLLtSolverInvertData : public MatrixInvertData
    {
    public :
        CompressedRowSparseMatrix<Real> Mfiltered;
        taucsmt_ccs_matrix<Real> matrix_taucs; //use only pointeur of Mfiltered!
        helper::vector<Real> B;
// 	    helper::vector<double> R;

        int* perm; //premutation
        int* invperm; //premutation inverse
        taucsmt_ccs_matrix<Real> * L; //factorization
        taucsmt_ccs_matrix<Real> * PAPT; //reordered matrix

        SparseTAUCSLLtSolverInvertData()
        {
            perm    = NULL;
            invperm = NULL;
            L       = NULL;
            PAPT    = NULL;
        }

        ~SparseTAUCSLLtSolverInvertData()
        {
            if (perm) free(perm);
            if (invperm) free(invperm);
            if (L) taucsmt_ccs_free(L);
            if (PAPT) taucsmt_ccs_free(PAPT);
            L = NULL;
            perm    = NULL;
            invperm = NULL;
            PAPT    = NULL;
        }
    };
};

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
