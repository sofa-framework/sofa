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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLDLSolver_INL

#include <sofa/component/linearsolver/SparseLDLSolver.h>
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
        for (int p = data->Lp [j] ; p < data->Lp[j+1] ; p++)
        {
            B [data->Li [p]] -= data->Lx[p] * B[j] ;
        }
    }
    for (int j = 0 ; j < data->n ; j++)
    {
        B [j] /= data->D[j] ;
    }
    for (int j = data->n-1 ; j >= 0 ; j--)
    {
        for (int p = data->Lp[j] ; p < data->Lp[j+1] ; p++)
        {
            B [j] -= data->Lx[p] * B [data->Li[p]] ;
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
    data->Mfiltered.resize(M.rowSize(),M.colSize());
    data->Mfiltered.copyNonZeros(M);
    data->Mfiltered.compress();

    data->colptr = (int *) &data->Mfiltered.getRowBegin()[0];
    data->rowind = (int *) &data->Mfiltered.getColsIndex()[0];
    data->values = (Real *) &data->Mfiltered.getColsValue()[0];

    B.resize(data->n);

    LDL_ordering(M);

    data->Lp.resize(data->n+1);
    LDL_symbolic(M) ;

    data->D.resize(data->n);
    data->Lx.resize(data->Lp[data->n]);
    data->Li.resize(data->Lp[data->n]);
    LDL_numeric(M) ;

}


template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::LDL_ordering(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);
    int  n,i,j,ip;
    int  num_flag     = 0;
    int  options_flag = 0;

    n   = data->n;
    adj.clear();
    xadj.resize(n+1);
    adj.resize(data->colptr[n]*2);
    data->perm.clear();
    data->invperm.clear();
    data->perm.resize(data->n);
    data->invperm.resize(data->n);

    for (j=0; j<n; j++)
    {
        for (ip = data->colptr[j]; ip < data->colptr[j+1]; ip++)
        {
            i = data->rowind[ip];
            if (i > j)
            {
                data->perm[i]++;
                data->perm[j]++;
            }
        }
    }

    xadj[0] = 0;
    for (i=1; i<=n; i++) xadj[i] = xadj[i-1] + data->perm[i-1];
    for (i=0; i<n; i++) data->perm[i] = xadj[i];

    for (j=0; j<n; j++)
    {
        for (ip = data->colptr[j]; ip < data->colptr[j+1]; ip++)
        {
            i = data->rowind[ip];
            if (i > j)
            {
                adj[data->perm[i]] = j;
                adj[data->perm[j]] = i;
                data->perm[i] ++;
                data->perm[j] ++;
            }
        }
    }

    METIS_NodeND(&n, &xadj[0],&adj[0], &num_flag, &options_flag, &data->perm[0],&data->invperm[0]);

}


template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::LDL_symbolic (Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);
    int i, p, kk ;

    Parent.clear();
    Lnz.clear();
    Flag.clear();
    Pattern.clear();
    Parent.resize(data->n);
    Lnz.resize(data->n);
    Flag.resize(data->n);
    Pattern.resize(data->n);

    for (int k = 0 ; k < data->n ; k++)
    {
        Parent [k] = -1 ;	    /* parent of k is not yet known */
        Flag [k] = k ;		    /* mark node k as visited */
        Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
        kk = data->perm[k];  /* kth original, or permuted, column */
        for (p = data->colptr[kk] ; p < data->colptr[kk+1] ; p++)
        {
            /* A (i,k) is nonzero (original or permuted A) */
            i = data->invperm[data->rowind[p]];
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

    data->Lp[0] = 0 ;
    for (int k = 0 ; k < data->n ; k++)
    {
        data->Lp [k+1] = data->Lp [k] + Lnz [k] ;
    }
}

template<class TMatrix, class TVector>
int SparseLDLSolver<TMatrix,TVector>::LDL_numeric(Matrix& M)
{
    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);
    double yi, l_ki ;
    int i, p, kk, len, top ;

    Y.resize(data->n);

    for (int k = 0 ; k < data->n ; k++)
    {
        Y [k] = 0.0 ;		    /* Y(0:k) is now all zero */
        top = data->n ;		    /* stack for pattern is empty */
        Flag [k] = k ;		    /* mark node k as visited */
        Lnz [k] = 0 ;		    /* count of nonzeros in column k of L */
        kk = data->perm[k];  /* kth original, or permuted, column */
        for (p = data->colptr[kk] ; p < data->colptr[kk+1] ; p++)
        {
            i = data->invperm[data->rowind[p]];	/* get A(i,k) */
            if (i <= k)
            {
                Y[i] += data->values[p] ;  /* scatter A(i,k) into Y (sum duplicates) */
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
            for (p = data->Lp[i] ; p < data->Lp[i] + Lnz [i] ; p++)
            {
                Y[data->Li[p]] -= data->Lx[p] * yi ;
            }
            l_ki = yi / data->D[i] ;	    /* the nonzero entry L(k,i) */
            data->D[k] -= l_ki * yi ;
            data->Li[p] = k ;	    /* store L(k,i) in column form of L */
            data->Lx[p] = l_ki ;
            Lnz[i]++ ;		    /* increment count of nonzeros in col i */
        }
        if (data->D[k] == 0.0) return (k) ;	    /* failure, D(k,k) is zero */
    }

    return (data->n) ;	/* success, diagonal of D is all nonzero */
}



} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
