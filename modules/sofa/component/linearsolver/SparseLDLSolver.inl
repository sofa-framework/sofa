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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL

#include <sofa/component/linearsolver/SparseLDLSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.inl>

extern "C" {
#include <metis.h>
}

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
SparseLDLSolver<TMatrix,TVector>::SparseLDLSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
{
}

template<class TMatrix, class TVector>
SparseLDLSolver<TMatrix,TVector>::~SparseLDLSolver() {}


template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    // permutation according to metis
    for (int i=0; i<data->n; i++) B[i] = r[data->perm[i]];

    for (int j = 0 ; j < data->n ; j++)
    {
        for (int p = data->colptr [j] ; p < data->colptr[j+1] ; p++)
        {
            B [data->rowind [p]] -= data->values[p] * B[j] ;
        }
    }
    for (int j = 0 ; j < data->n ; j++)
    {
        B [j] /= data->D[j] ;
    }
    for (int j = data->n-1 ; j >= 0 ; j--)
    {
        for (int p = data->colptr[j] ; p < data->colptr[j+1] ; p++)
        {
            B [j] -= data->values[p] * B [data->rowind[p]] ;
        }
    }

    for (int i=0; i<data->n; i++) z[i] = B[data->invperm[i]];
}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::invert(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    //remplir A avec M
    data->n = M.colSize();// number of columns
    data->Mfiltered.clear();
    data->Mfiltered.resize(M.rowSize(),M.colSize());
    data->Mfiltered.copyNonZeros(M);
    data->Mfiltered.compress();

    data->Mcolptr = (int *) &data->Mfiltered.getRowBegin()[0];
    data->Mrowind = (int *) &data->Mfiltered.getColsIndex()[0];
    data->Mvalues = (Real *) &data->Mfiltered.getColsValue()[0];

    LDL_ordering(M);
    LDL_symbolic(M);
    LDL_numeric(M);
}


template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::LDL_ordering(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);
    int  num_flag     = 0;
    int  options_flag = 0;

    xadj.resize(data->n+1);
    adj.resize(data->Mcolptr[data->n]-data->n);
    data->perm.resize(data->n);
    data->invperm.resize(data->n);
    B.resize(data->n);

    int it = 0;
    for (int j=0; j<data->n; j++)
    {
        xadj[j] = data->Mcolptr[j] - j;

        for (int ip = data->Mcolptr[j]; ip < data->Mcolptr[j+1]; ip++)
        {
            int i = data->Mrowind[ip];
            if (i != j) adj[it++] = i;
        }
    }
    xadj[data->n] = data->Mcolptr[data->n] - data->n;

    METIS_NodeND(&data->n, &xadj[0],&adj[0], &num_flag, &options_flag, &data->perm[0],&data->invperm[0]);
}


template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::LDL_symbolic (Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    Parent.clear();
    Lnz.clear();
    Flag.clear();
    Pattern.clear();
    Parent.resize(data->n);
    Lnz.resize(data->n);
    Flag.resize(data->n);
    Pattern.resize(data->n);
    data->colptr.resize(data->n+1);

    for (int k = 0 ; k < data->n ; k++)
    {
        Parent [k] = -1 ;	    /* parent of k is not yet known */
        Flag [k] = k ;		    /* mark node k as visited */
        Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
        int kk = data->perm[k];  /* kth original, or permuted, column */
        for (int p = data->Mcolptr[kk] ; p < data->Mcolptr[kk+1] ; p++)
        {
            /* A (i,k) is nonzero (original or permuted A) */
            int i = data->invperm[data->Mrowind[p]];
            if (i < k)
            {
                /* follow path from i to root of etree, stop at flagged node */
                for ( ; Flag [i] != k ; i = Parent [i])
                {
                    /* find parent of i if not yet determined */
                    if (Parent [i] == -1) Parent [i] = k ;
                    Lnz [i]++ ;				/* L (k,i) is nonzero */
                    Flag [i] = k ;			/* mark i as visited */
                }
            }
        }
    }

    data->colptr[0] = 0 ;
    for (int k = 0 ; k < data->n ; k++)
    {
        data->colptr [k+1] = data->colptr [k] + Lnz [k] ;
    }
}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::LDL_numeric(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);
    Real yi, l_ki ;
    int i, p, kk, len, top ;

    Y.resize(data->n);
    data->D.resize(data->n);
    data->values.resize(data->colptr[data->n]);
    data->rowind.resize(data->colptr[data->n]);

    for (int k = 0 ; k < data->n ; k++)
    {
        Y [k] = 0.0 ;		    /* Y(0:k) is now all zero */
        top = data->n ;		    /* stack for pattern is empty */
        Flag [k] = k ;		    /* mark node k as visited */
        Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
        kk = data->perm[k];  /* kth original, or permuted, column */
        for (p = data->Mcolptr[kk] ; p < data->Mcolptr[kk+1] ; p++)
        {
            i = data->invperm[data->Mrowind[p]];	/* get A(i,k) */
            if (i <= k)
            {
                Y[i] += data->Mvalues[p] ;  /* scatter A(i,k) into Y (sum duplicates) */
                for (len = 0 ; Flag[i] != k ; i = Parent[i])
                {
                    Pattern [len++] = i ;   /* L(k,i) is nonzero */
                    Flag [i] = k ;	    /* mark i as visited */
                }
                while (len > 0) Pattern[--top] = Pattern [--len] ;
            }
        }
        /* compute numerical values kth row of L (a sparse triangular solve) */
        data->D[k] = Y [k] ;		    /* get D(k,k) and clear Y(k) */
        Y[k] = 0.0 ;
        for ( ; top < data->n ; top++)
        {
            i = Pattern [top] ;	    /* Pattern [top:n-1] is pattern of L(:,k) */
            yi = Y [i] ;	    /* get and clear Y(i) */
            Y [i] = 0.0 ;
            for (p = data->colptr[i] ; p < data->colptr[i] + Lnz [i] ; p++)
            {
                Y[data->rowind[p]] -= data->values[p] * yi ;
            }
            l_ki = yi / data->D[i] ;	    /* the nonzero entry L(k,i) */
            data->D[k] -= l_ki * yi ;
            data->rowind[p] = k ;	    /* store L(k,i) in column form of L */
            data->values[p] = l_ki ;
            Lnz[i]++ ;		    /* increment count of nonzeros in col i */
        }
        if (data->D[k] == 0.0)
        {
            std::cerr << "SparseLDLSolver failure to factorize, D(k,k) is zero" << std::endl;
            return;
        }
    }
}



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
