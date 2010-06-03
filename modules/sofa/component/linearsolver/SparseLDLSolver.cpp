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
#include <sofa/component/linearsolver/SparseLDLSolver.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>

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
void SparseLDLSolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& z, Vector& r)
{
    z = r;
    ldl_lsolve (n, z.ptr(), &Lp[0], &Li[0], &Lx[0]) ;
    ldl_dsolve (n, z.ptr(), &D[0]) ;
    ldl_ltsolve (n, z.ptr(), &Lp[0], &Li[0], &Lx[0]) ;

}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::invert(Matrix& M)
{
    M.compress();

    //remplir A avec M
    n = M.colBSize();// number of columns

    A_p = M.getRowBegin();
    A_i = M.getColsIndex();
    A_x = M.getColsValue();

    D.resize(n);
    Y.resize(n);
    Lp.resize(n+1);
    Parent.resize(n);
    Lnz.resize(n);
    Flag.resize(n);
    Pattern.resize(n);

    ldl_symbolic (n, &A_p[0], &A_i[0], &Lp[0], &Parent[0], &Lnz[0], &Flag[0], NULL, NULL) ;

    Lx.resize(Lp[n]);
    Li.resize(Lp[n]);
    ldl_numeric (n, &A_p[0], &A_i[0], &A_x[0], &Lp[0], &Parent[0], &Lnz[0], &Li[0], &Lx[0], &D[0], &Y[0], &Pattern[0], &Flag[0], NULL, NULL) ;

}


SOFA_DECL_CLASS(SparseLDLSolver)

int SparseLDLSolverClass = core::RegisterObject("Direct linear solver based on Sparse LDL^T factorization, implemented with the CSPARSE library")
        .add< SparseLDLSolver< CompressedRowSparseMatrix<double>,FullVector<double> > >(true)
        ;

template class SOFA_COMPONENT_LINEARSOLVER_API SparseLDLSolver< CompressedRowSparseMatrix<double>,FullVector<double> >;

} // namespace linearsolver

} // namespace component

} // namespace sofa
