/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2015 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_SparseLUSolver_INL
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include <SofaSparseSolver/SparseLUSolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <csparse.h>
//#include<dartPlugin/core/AdaptiveTopologyManager.h>

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

template<class TMatrix, class TVector,class TThreadManager>
SparseLUSolver<TMatrix,TVector,TThreadManager>::SparseLUSolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_tol( initData(&f_tol,0.001,"tolerance","tolerance of factorization") )
    , f_filterValue(initData(&f_filterValue, 0.0,"filterValue","Smaller value considered in the factorization"))
{
}


template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::solve (Matrix& M, Vector& z, Vector& r)// solve Mz=r
{
    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::solve"<<std::endl;
    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::solve r = "<<r<<std::endl;
    SpaseLUInvertData<Real> * invertData = (SpaseLUInvertData<Real>*) this->getMatrixInvertData(&M);/// use invertData previously created by invert method
    int n = invertData->A.n;

    cs_ipvec (n, invertData->N->Pinv, r.ptr(), invertData->tmp) ; // tmp(Pinv)=r     	/* x = P*b */
    cs_lsolve (invertData->N->L, invertData->tmp) ; // solve L*tmp = tmp           		/* x = L\x */
    cs_usolve (invertData->N->U, invertData->tmp) ;// solve U*tmp = tmp         		/* x = U\x */
    cs_ipvec (n, invertData->S->Q, invertData->tmp, z.ptr()) ; // z(Q) = tmp            /* b = Q*x */

    // s in PCG should be z here
    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::solve z = "<<z<<std::endl;
}

//template<class TMatrix, class TVector,class TThreadManager>
//void SparseLUSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
//{
//     //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert"<<std::endl;
//    SpaseLUInvertData<Real> * invertData = (SpaseLUInvertData<Real>*) this->getMatrixInvertData(&M);
//    int order = -1; //?????

//    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M.getRowBegin() = "<<M.getRowBegin()<<std::endl;

////    std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M(0,0) = "<<M.element(0,0)<<std::endl;
////    for(unsigned i=0; i< M.rowSize(); ++i)
////    {
////        std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M(i,i) = "<<M.element(i,i)<<std::endl;
////    }

//    if (invertData->S) cs_sfree(invertData->S);
//    if (invertData->N) cs_nfree(invertData->N);
//    if (invertData->tmp) cs_free(invertData->tmp);
//    M.compress();
//    //remplir A avec M
//    invertData->A.nzmax = M.getColsValue().size();	// maximum number of entries
//    invertData->A.m = M.rowBSize();					// number of rows
//    invertData->A.n = M.colBSize();					// number of columns
//    invertData->A_p = M.getRowBegin();

//    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M.colBSize() = "<<M.colBSize()<<std::endl;

//    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, invertData->A_p size = "<<invertData->A_p.size()<<std::endl;

//    invertData->A.p = (int *) &(invertData->A_p[0]);							// column pointers (size n+1) or col indices (size nzmax)
//    invertData->A_i = M.getColsIndex();
//    invertData->A.i = (int *) &(invertData->A_i[0]);							// row indices, size nzmax
//    invertData->A_x = M.getColsValue();
//    invertData->A.x = (Real *) &(invertData->A_x[0]);				// numerical values, size nzmax
//    invertData->A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
//    cs_dropzeros( &invertData->A );

//    //M.check_matrix();
//    //CompressedRowSparseMatrix<double>::check_matrix(-1 /*A.nzmax*/,A.m,A.n,A.p,A.i,A.x);
//    //sout << "diag =";
//    //for (int i=0;i<A.n;++i) sout << " " << M.element(i,i);
//    //sout << sendl;
//    //sout << "SparseCholeskySolver: start factorization, n = " << A.n << " nnz = " << A.p[A.n] << sendl;
//    invertData->tmp = (Real *) cs_malloc (invertData->A.n, sizeof (Real)) ;
//    invertData->S = cs_sqr (&invertData->A, order, 0) ;		/* ordering and symbolic analysis */
//    invertData->N = cs_lu (&invertData->A, invertData->S, f_tol.getValue()) ;		/* numeric LU factorization */

////    double pcl = (100.0*invertData->N->L->p[invertData->N->L->n])/(double) (invertData->A.n*invertData->A.n);
////    double pcu = (100.0*invertData->N->U->p[invertData->N->U->n])/(double) (invertData->A.n*invertData->A.n);
////    double pca = (100.0*invertData->A.p[invertData->A.n])/(double) (invertData->A.n*invertData->A.n);
////    printf("n=%d m=%d = Lnz=%d Unz=%d Anz=%d Lpc=%f Upc=%f Apc=%f\n",
////           invertData->A.n,invertData->A.m,
////           invertData->N->L->p[invertData->N->L->n],
////           invertData->N->U->p[invertData->N->U->n],
////           invertData->A.p[invertData->A.n],
////           pcl,pcu,pca);



////    int sz = invertData->A.p[invertData->A.n] ;	    /* column pointers (size n+1) or col indices (size nzmax) */
////    for(unsigned i=0; i<sz; i++)
////    {
////        std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert A.x = "<<invertData->A.x[i]<<std::endl;
////    }

//    /// print L

////    std::cout<<"L, size = "<<invertData->N->L->n <<std::endl;
////    for(unsigned i=0; i< invertData->N->L->n; ++i)// n: number of column
////    {
////        for(unsigned k=invertData->N->L->p[i]; k<invertData->N->L->p[i+1]; ++k)
////        {
////            // row col val
////            std::cout<<"L("<<invertData->N->L->i[k]+1<<","<<i+1<<")= "<<invertData->N->L->x[k]<<";"<<std::endl;
////            //printf("%d, %d, %f\n",invertData->N->L->i[k],i,invertData->N->L->x[k]);
////        }
////    }

//    /// print A
//    std::cout<<"A, size = "<<invertData->A.n <<std::endl;
//    const Real ZERO = 1e-10;
//    for(unsigned i=0; i< invertData->A.n; ++i)// n: number of column
//    {
//        for(unsigned k=invertData->A.p[i]; k<invertData->A.p[i+1]; ++k)
//        {
//            // row col val
//            if(fabs(invertData->A.x[k])<ZERO)
//                std::cout<<"A("<<invertData->A.i[k]+1<<","<<i+1<<")= "<<0<<";"<<std::endl;
//            else
//            std::cout<<"A("<<invertData->A.i[k]+1<<","<<i+1<<")= "<<invertData->A.x[k]<<";"<<std::endl;
//            //printf("%d, %d, %f\n",invertData->N->L->i[k],i,invertData->N->L->x[k]);

//        }
//    }


//    //sout << "SparseCholeskySolver: factorization complete, nnz = " << N->L->p[N->L->n] << sendl;
//}





template<class TMatrix, class TVector,class TThreadManager>
void SparseLUSolver<TMatrix,TVector,TThreadManager>::invert(Matrix& M)
{
     //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert"<<std::endl;
    SpaseLUInvertData<Real> * invertData = (SpaseLUInvertData<Real>*) this->getMatrixInvertData(&M);///<create invertData: factorization LU of matrix M


    //std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M.getRowBegin() = "<<M.getRowBegin()<<std::endl;

//    std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M(0,0) = "<<M.element(0,0)<<std::endl;
//    for(unsigned i=0; i< M.rowSize(); ++i)
//    {
//        std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert, M(i,i) = "<<M.element(i,i)<<std::endl;
//    }

    //Mfiltered.copyNonZeros(M);
    Mfiltered.copyNonSmall(M,f_filterValue.getValue());
    Mfiltered.compress();

    int n = M.colSize();

    int * M_colptr = (int *) &Mfiltered.getRowBegin()[0];
    int * M_rowind = (int *) &Mfiltered.getColsIndex()[0];
    Real * M_values = (Real *) &Mfiltered.getColsValue()[0];


    if (invertData->S) cs_sfree(invertData->S);
    if (invertData->N) cs_nfree(invertData->N);
    if (invertData->tmp) cs_free(invertData->tmp);

    //remplir A avec M
    invertData->A.nzmax = M_colptr[n];	// maximum number of entries
    invertData->A.m = n;					// number of rows
    invertData->A.n = n;					// number of columns
    invertData->A.p =  M_colptr;// column pointers (size n+1) or col indices (size nzmax)

    invertData->A.i = M_rowind;// row indices, size nzmax

    invertData->A.x = M_values;// numerical values, size nzmax

    invertData->A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &invertData->A );

    //M.check_matrix();
    //CompressedRowSparseMatrix<double>::check_matrix(-1 /*A.nzmax*/,A.m,A.n,A.p,A.i,A.x);
    //sout << "diag =";
    //for (int i=0;i<A.n;++i) sout << " " << M.element(i,i);
    //sout << sendl;
    //sout << "SparseCholeskySolver: start factorization, n = " << A.n << " nnz = " << A.p[A.n] << sendl;
    invertData->tmp = (Real *) cs_malloc (invertData->A.n, sizeof (Real)) ;

    //ordering (permutation) by metis
    invertData->perm.clear();invertData->perm.fastResize(invertData->A.n);
    invertData->invperm.clear();invertData->invperm.fastResize(invertData->A.n);
    LU_ordering(invertData->A.n,M_colptr,M_rowind,&invertData->perm[0],&invertData->invperm[0]);

    //int order=-1;
    //invertData->S = cs_sqr (&invertData->A, order, 0) ;//S: symbolic		/* ordering and symbolic analysis */
    invertData->S = CSPARSE_sqr (&invertData->A, &invertData->perm[0]) ;//S: symbolic		/* ordering and symbolic analysis */

    //invertData->N = cs_lu (&invertData->A, invertData->S, f_tol.getValue()) ;// N: numeric		/* numeric LU factorization */
    invertData->N = CSPARSE_lu (&invertData->A, invertData->S, f_tol.getValue()) ;// N: numeric		/* numeric LU factorization */


//    double pcl = (100.0*invertData->N->L->p[invertData->N->L->n])/(double) (invertData->A.n*invertData->A.n);
//    double pcu = (100.0*invertData->N->U->p[invertData->N->U->n])/(double) (invertData->A.n*invertData->A.n);
//    double pca = (100.0*invertData->A.p[invertData->A.n])/(double) (invertData->A.n*invertData->A.n);
//    printf("n=%d m=%d = Lnz=%d Unz=%d Anz=%d Lpc=%f Upc=%f Apc=%f\n",
//           invertData->A.n,invertData->A.m,
//           invertData->N->L->p[invertData->N->L->n],
//           invertData->N->U->p[invertData->N->U->n],
//           invertData->A.p[invertData->A.n],
//           pcl,pcu,pca);



//    int sz = invertData->A.p[invertData->A.n] ;	    /* column pointers (size n+1) or col indices (size nzmax) */
//    for(unsigned i=0; i<sz; i++)
//    {
//        std::cout<<"SparseLUSolver<TMatrix,TVector,TThreadManager>::invert A.x = "<<invertData->A.x[i]<<std::endl;
//    }

    /// print L

//    std::cout<<"L, size = "<<invertData->N->L->n <<std::endl;
//    for(unsigned i=0; i< invertData->N->L->n; ++i)// n: number of column
//    {
//        for(unsigned k=invertData->N->L->p[i]; k<invertData->N->L->p[i+1]; ++k)
//        {
//            // row col val
//            std::cout<<"L("<<invertData->N->L->i[k]+1<<","<<i+1<<")= "<<invertData->N->L->x[k]<<";"<<std::endl;
//            //printf("%d, %d, %f\n",invertData->N->L->i[k],i,invertData->N->L->x[k]);
//        }
//    }

    /// print A
//    std::cout<<"A, size = "<<invertData->A.n <<std::endl;
//    //const Real ZERO = 1e-10;
//    for(unsigned i=0; i< invertData->A.n; ++i)// n: number of column
//    {
//        for(unsigned k=invertData->A.p[i]; k<invertData->A.p[i+1]; ++k)
//        {
//            // row col val
////            if(fabs(invertData->A.x[k])<ZERO)
////                std::cout<<"A("<<invertData->A.i[k]+1<<","<<i+1<<")= "<<0<<";"<<std::endl;
////            else
//            std::cout<<"A("<<invertData->A.i[k]+1<<","<<i+1<<")= "<<invertData->A.x[k]<<";"<<std::endl;
//            //printf("%d, %d, %f\n",invertData->N->L->i[k],i,invertData->N->L->x[k]);

//        }
//    }


    //sout << "SparseCholeskySolver: factorization complete, nnz = " << N->L->p[N->L->n] << sendl;
}




} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
