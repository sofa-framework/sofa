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
#include <sofa/component/linearsolver/JacobiPreconditioner.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/DiagonalMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
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
JacobiPreconditioner<TMatrix,TVector>::JacobiPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
}

/// Solve P^-1 Mx= P^-1 b
// P[i][j] = M[i][j] ssi i=j
//P^-1[i][j] = 1/M[i][j]
template<class TMatrix, class TVector>
void JacobiPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    for (unsigned i=0; i<z.size(); i++) z.set(i,r.element(i) / M.element(i,i)); //si i==j;
}

SOFA_DECL_CLASS(JacobiPreconditioner)

int JacobiPreconditionerClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
//.add< JacobiPreconditioner<GraphScatteredMatrix,GraphScatteredVector> >(true)
        .add< JacobiPreconditioner< SparseMatrix<double>, FullVector<double> > >()
        .add< JacobiPreconditioner<NewMatBandMatrix,NewMatVector> >()
        .add< JacobiPreconditioner<DiagonalMatrix<double>,FullVector<double> > >()
        .add< JacobiPreconditioner<DiagonalMatrix<float>,FullVector<float> > >(true)
        .add< JacobiPreconditioner<NewMatMatrix,NewMatVector> >()
        .add< JacobiPreconditioner<NewMatSymmetricMatrix,NewMatVector> >()
        .add< JacobiPreconditioner<NewMatSymmetricBandMatrix,NewMatVector> >()
        .add< JacobiPreconditioner< FullMatrix<double>, FullVector<double> > >()
        .addAlias("JCGSolver")
        .addAlias("JConjugateGradient")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

