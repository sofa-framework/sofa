/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#include <sofa/component/linearsolver/SSORPreconditioner.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/helper/system/thread/CTime.h>


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
SSORPreconditioner<TMatrix,TVector>::SSORPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
}

/*
// solve (D+U) * D^-1 * (D+L)
template<class TMatrix, class TVector>
void SSORPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r) {
	double t2 = CTime::getRefTime();

	//Solve (D+U) * u3 = r;
	for (int j=M.rowSize()-1;j>=0;j--) {
		double temp = 0.0;
		for (unsigned i=j+1;i<M.rowSize();i++) {
			temp += u3[i] * M.element(i,j);
		}
		u3[j] = (r[j] - temp) / M.element(j,j);
	}

	//Solve D-1 * u2 = u3;
	for (unsigned j=0;j<M.rowSize();j++) {
		u2[j] = u3[j] * M.element(j,j);
	}

	//Solve (L+D) * z = u2
	for (unsigned j=0;j<M.rowSize();j++) {
		double temp = 0.0;
		for (unsigned i=0;i<j;i++) {
			temp += z[i] * M.element(i,j);
		}
		z[j] = (u2[j] - temp) / M.element(j,j);
	}

	printf("%f ",(CTime::getRefTime() - t2) / (double)CTime::getRefTicksPerSec());
}
*/

// solve (D+U) * ( I + D^-1 * U)
template<class TMatrix, class TVector>
void SSORPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    //double t2 = CTime::getRefTime();

    //Solve (D+U) * u3 = r;
    for (int j=M.rowSize()-1; j>=0; j--)
    {
        double temp = 0.0;
        for (unsigned i=j+1; i<M.rowSize(); i++)
        {
            temp += z[i] * M.element(i,j);
        }
        z[j] = (r[j] - temp) / M.element(j,j);
    }

    //Solve (I + D^-1 * L) * z = u3
    for (unsigned j=0; j<M.rowSize(); j++)
    {
        double temp = 0.0;
        for (unsigned i=0; i<j; i++)
        {
            temp += z[i] * M.element(i,j) / M.element(j,j);
        }
        z[j] = z[j] - temp;
        // we can reuse z because all values that we read are updated
    }

}

template<>
void SSORPreconditioner<SparseMatrix<double>, FullVector<double> >::solve (Matrix& M, Vector& z, Vector& r)
{
    int n = M.rowSize();

    //Solve (D+U) * t = r;
    for (int j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        for (Matrix::LElementConstIterator it = ++M[j].find(j), end = M[j].end(); it != end; ++it)
        {
            int i = it->first;
            double e = it->second;
            temp += z[i] * e;
        }
        z[j] = (r[j] - temp) * inv_diag[j];
    }

    //Solve (I + D^-1 * L) * z = t
    for (int j=0; j<n; j++)
    {
        double temp = 0.0;
        for (Matrix::LElementConstIterator it = M[j].begin(), end = M[j].find(j); it != end; ++it)
        {
            int i = it->first;
            double e = it->second;
            temp += z[i] * e;
        }
        z[j] -= temp * inv_diag[j];
        // we can reuse z because all values that we read are updated
    }
}

template<>
void SSORPreconditioner<CompressedRowSparseMatrix<double>, FullVector<double> >::solve (Matrix& M, Vector& z, Vector& r)
{
    int n = M.rowSize();
    //const Matrix::VecIndex& rowIndex = M.getRowIndex();
    const Matrix::VecIndex& colsIndex = M.getColsIndex();
    const Matrix::VecBloc& colsValue = M.getColsValue();
    //Solve (D+U) * t = r;
    for (int j=n-1; j>=0; j--)
    {
        double temp = 0.0;
        Matrix::Range rowRange = M.getRowRange(j);
        int xi = rowRange.begin();
        while (xi < rowRange.end() && colsIndex[xi] <= j) ++xi;
        for (; xi < rowRange.end(); ++xi)
        {
            int i = colsIndex[xi];
            double e = colsValue[xi];
            temp += z[i] * e;
        }
        z[j] = (r[j] - temp) * inv_diag[j];
    }

    //Solve (I + D^-1 * L) * z = t
    for (int j=0; j<n; j++)
    {
        double temp = 0.0;
        Matrix::Range rowRange = M.getRowRange(j);
        int xi = rowRange.begin();
        for (; xi < rowRange.end() && colsIndex[xi] < j; ++xi)
        {
            int i = colsIndex[xi];
            double e = colsValue[xi];
            temp += z[i] * e;
        }
        z[j] -= temp * inv_diag[j];
        // we can reuse z because all values that we read are updated
    }
}
template<class TMatrix, class TVector>
void SSORPreconditioner<TMatrix,TVector>::invert(Matrix& M)
{
    int n = M.rowSize();
    inv_diag.resize(n);
    for (int j=0; j<n; j++) inv_diag[j] = 1.0 / M.element(j,j);
}

SOFA_DECL_CLASS(SSORPreconditioner)

int SSORPreconditionerClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
//.add< SSORPreconditioner<GraphScatteredMatrix,GraphScatteredVector> >(true)
        .add< SSORPreconditioner< SparseMatrix<double>, FullVector<double> > >(true)
        .add< SSORPreconditioner< CompressedRowSparseMatrix<double>, FullVector<double> > >()
//.add< SSORPreconditioner<NewMatBandMatrix,NewMatVector> >(true)
//.add< SSORPreconditioner<NewMatMatrix,NewMatVector> >()
        .add< SSORPreconditioner<NewMatSymmetricMatrix,NewMatVector> >()
//.add< SSORPreconditioner<NewMatSymmetricBandMatrix,NewMatVector> >()
        .add< SSORPreconditioner< FullMatrix<double>, FullVector<double> > >()
        .addAlias("SSORSolver")
        .addAlias("SSORConjugateGradient")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

