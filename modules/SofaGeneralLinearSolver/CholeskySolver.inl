/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_LINEARSOLVER_CholeskySolver_INL
#define SOFA_COMPONENT_LINEARSOLVER_CholeskySolver_INL

#include <SofaGeneralLinearSolver/CholeskySolver.h>
#include <sofa/core/visual/VisualParams.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/behavior/LinearSolver.h>
#include <math.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{

template<class TMatrix, class TVector>
CholeskySolver<TMatrix,TVector>::CholeskySolver()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
{
}


/// Factorisation : A = LL^t
/// A x = b <=> LL^t x = b
///        <=> L u = b , L^t x = u
template<class TMatrix, class TVector>
void CholeskySolver<TMatrix,TVector>::solve (Matrix& /*M*/, Vector& z, Vector& r)
{
    //Compute L
    int n = L.colSize();

    //Solve L u = b
    for (int j=0; j<n; j++)
    {
        double temp = 0.0;
        double d = 1.0 / L.element(j,j);
        for (int i=0; i<j; i++)
        {
            temp += z[i] * L.element(i,j);
        }
        z[j] = (Real)((r[j] - temp) * d);
    }

    //Solve L^t x = u
    for (int j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        double d = 1.0 / L.element(j,j);
        for (int i=j+1; i<n; i++)
        {
            temp += z[i] * L.element(j,i);
        }
        z[j] = (Real)((z[j] - temp) * d);
    }
}

template<class TMatrix, class TVector>
void CholeskySolver<TMatrix,TVector>::invert(Matrix& M)
{
    int n = M.colSize();
    double ss,d;

    L.resize(n,n);
    if( M.element(0,0) <= 0 ) serr<<"CholeskySolver<TMatrix,TVector>::invert, matrix is not positive definite " << sendl;
    d = 1.0 / sqrt(M.element(0,0));
    for (int i=0; i<n; i++)
    {
        L.set(0,i,M.element(i,0) * d);
    }

    for (int j=1; j<n; j++)
    {
        ss=0;
        for (int k=0; k<j; k++) ss+=L.element(k,j)*L.element(k,j);

        if( M.element(j,j)-ss <= 0 ) serr<<"CholeskySolver<TMatrix,TVector>::invert, matrix is not positive definite " << sendl;
        d = 1.0 / sqrt(M.element(j,j)-ss);
        L.set(j,j,(M.element(j,j)-ss) * d);


        for (int i=j+1; i<n; i++)
        {
            ss=0;
            for (int k=0; k<j; k++) ss+=L.element(k,i)*L.element(k,j);
            L.set(j,i,(M.element(i,j)-ss) * d);
        }
    }
}

/*
template<class TMatrix, class TVector>
void CholeskySolver<TMatrix,TVector>::invert(Matrix& M) {
	int n = M.colSize();
	L.resize(n,n);
	double somme;

	L.set(0,0,sqrt(M.element(0,0)));

	for (int i=1; i<n; i++) {
		for (int j=0; j<i; j++) {
			somme = 0.0;

			for (int k=0; k<j; k++) {
				somme = somme + L.element(i,k)*L.element(j,k);
			}
			L.set(i,j,(M.element(i,j)-somme) / L.element(j,j));
		}

		somme = 0.0;
		for (int k=0; k<i; k++) {
			somme = somme + L.element(i,k)*L.element(i,k);
		}
		L.set(i,i,sqrt(M.element(i,i)-somme));
	}
}
*/

} // namespace linearsolver

} // namespace component

} // namespace sofa

#endif
