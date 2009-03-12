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
#include <sofa/component/linearsolver/SparseCholeskySolver.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::simulation;
using namespace sofa::core::objectmodel;
using sofa::helper::system::thread::CTime;
using sofa::helper::system::thread::ctime_t;
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::SparseCholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
    , S(NULL), N(NULL), tmp(NULL)
{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
}

template<class TMatrix, class TVector>
SparseCholeskySolver<TMatrix,TVector>::~SparseCholeskySolver()
{
    if (S) cs_sfree (S);
    if (N) cs_nfree (N);
    if (tmp) cs_free (tmp);
}


template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& z, Vector& r)
{
    int n = A.n;

    cs_ipvec (n, S->Pinv, r.ptr(), tmp);	//x = P*b

    cs_lsolve (N->L, tmp);			//x = L\x
    cs_ltsolve (N->L, tmp);			//x = L'\x/

    cs_pvec (n, S->Pinv, tmp, z.ptr());	 //b = P'*x
}

template<class TMatrix, class TVector>
void SparseCholeskySolver<TMatrix,TVector>::invert(Matrix& M)
{
    int order = -1; //?????

    if (S) cs_sfree(S);
    if (N) cs_nfree(N);
    if (tmp) cs_free(tmp);
    M.compress();
    //remplir A avec M
    A.nzmax = M.getColsValue().size();	// maximum number of entries
    A.m = M.rowBSize();					// number of rows
    A.n = M.colBSize();					// number of columns
    A_p = M.getRowBegin();
    A.p = (int *) &(A_p[0]);							// column pointers (size n+1) or col indices (size nzmax)
    A_i = M.getColsIndex();
    A.i = (int *) &(A_i[0]);							// row indices, size nzmax
    A_x = M.getColsValue();
    A.x = (double *) &(A_x[0]);				// numerical values, size nzmax
    A.nz = -1;							// # of entries in triplet matrix, -1 for compressed-col
    cs_dropzeros( &A );

    //M.check_matrix();
    //CompressedRowSparseMatrix<double>::check_matrix(-1 /*A.nzmax*/,A.m,A.n,A.p,A.i,A.x);
    //sout << "diag =";
    //for (int i=0;i<A.n;++i) sout << " " << M.element(i,i);
    //sout << sendl;
    //sout << "SparseCholeskySolver: start factorization, n = " << A.n << " nnz = " << A.p[A.n] << sendl;
    tmp = (double *) cs_malloc (A.n, sizeof (double)) ;
    S = cs_schol (&A, order) ;		/* ordering and symbolic analysis */
    N = cs_chol (&A, S) ;		/* numeric Cholesky factorization */
    //sout << "SparseCholeskySolver: factorization complete, nnz = " << N->L->p[N->L->n] << sendl;
}

template<class TMatrix, class TVector>
bool SparseCholeskySolver<TMatrix,TVector>::readFile(std::istream& in)
{
    std::cout << "Read SparseCholeskySolver" << std::endl;

    std::string s;

    in >> s;

    if (! s.compare("SparseCholeskySolver"))
    {
        std::cout << "File not contain a SparseCholeskySolver" << std::endl;
        return false;
    }

    // read A
    in >> A.nzmax;
    in >> A.m;
    in >> A.n;

    A_x.read(in);
    A_i.read(in);
    A_p.read(in);

    in >> A.nz;

    // read Pinv
    //S->Pinv = (int *) cs_malloc(A.n,sizeof(int));
    //in.read((char *)S->Pinv,A.n * sizeof(int));

    //read N->L
    in >> N->L->nzmax;
    in >> N->L->m;
    in >> N->L->n;

    N->L->p = (int *) cs_malloc(N->L->n+1,sizeof(int));
    N->L->i = (int *) cs_malloc(N->L->nzmax,sizeof(int));
    N->L->x = (double *) cs_malloc(N->L->nzmax,sizeof(double));

    in.read((char *) N->L->p,(N->L->n+1) * sizeof(int));
    in.read((char *) N->L->i,N->L->nzmax * sizeof(int));
    in.read((char *) N->L->x,N->L->nzmax * sizeof(double));

    in >> N->L->nz;

    return true;
}

template<class TMatrix, class TVector>
bool SparseCholeskySolver<TMatrix,TVector>::writeFile(std::ostream& out)
{
    std::cout << "Write SparseCholeskySolver" << std::endl;

    std::string s = "SparseCholeskySolver";
    out << s << endl;

    // save A
    out << A.nzmax << endl;
    out << A.m << endl;
    out << A.n << endl;

    A_x.write(out);
    A_i.write(out);
    A_p.write(out);
    out << A.nz << endl;

    // save Pinv
    //out.write((char *) S->Pinv,A.n * sizeof(int));

    //save N->L
    out << N->L->nzmax << endl;
    out << N->L->m << endl;
    out << N->L->n << endl;
    out.write((char *) N->L->p,(N->L->n+1) * sizeof(int));
    out.write((char *) N->L->i,N->L->nzmax * sizeof(int));
    out.write((char *) N->L->x,N->L->nzmax * sizeof(double));
    out << N->L->nz << endl;

    return true;
}

SOFA_DECL_CLASS(SparseCholeskySolver)

int SparseCholeskySolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< SparseCholeskySolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        .addAlias("SparseCholeskySolverAlias")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

