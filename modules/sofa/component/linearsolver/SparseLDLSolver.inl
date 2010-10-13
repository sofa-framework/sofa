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
    z = r;

    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

    ldl_lsolve (data->n, &z[0], &data->Lp[0], &data->Li[0], &data->Lx[0]) ;
    ldl_dsolve (data->n, &z[0], &data->D[0]) ;
    ldl_ltsolve (data->n, &z[0], &data->Lp[0], &data->Li[0], &data->Lx[0]) ;

}

template<class TMatrix, class TVector>
void SparseLDLSolver<TMatrix,TVector>::invert(Matrix& M)
{
    M.compress();

    SparseLDLSolverInvertData * data = (SparseLDLSolverInvertData *) getMatrixInvertData(&M);

// 	  printf("element\n");
// 	  for (int j=0;j<12;j++) {
// 	    for (int i=0;i<12;i++) {
// 		printf("%f ",M.element(j,i));
// 	    }
// 	    printf("\n");
// 	  }

    //remplir A avec M
    data->n = M.colBSize();// number of columns

    data->A_p = M.getRowBegin();
    data->A_i = M.getColsIndex();
    data->A_x = M.getColsValue();

    data->D.resize(data->n);
    data->Y.resize(data->n);
    data->Lp.resize(data->n+1);
    data->Parent.resize(data->n);
    data->Lnz.resize(data->n);
    data->Flag.resize(data->n);
    data->Pattern.resize(data->n);

    ldl_symbolic (data->n, &data->A_p[0], &data->A_i[0], &data->Lp[0], &data->Parent[0], &data->Lnz[0], &data->Flag[0], NULL, NULL) ;

    data->Lx.resize(data->Lp[data->n]);
    data->Li.resize(data->Lp[data->n]);
    ldl_numeric (data->n, &data->A_p[0], &data->A_i[0], &data->A_x[0], &data->Lp[0], &data->Parent[0], &data->Lnz[0], &data->Li[0], &data->Lx[0], &data->D[0], &data->Y[0], &data->Pattern[0], &data->Flag[0], NULL, NULL) ;

}

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
