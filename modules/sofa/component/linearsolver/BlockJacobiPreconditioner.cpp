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
#include <sofa/component/linearsolver/BlockJacobiPreconditioner.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"
#include <sofa/core/objectmodel/BaseContext.h>
#include <sofa/core/componentmodel/behavior/LinearSolver.h>
#include <math.h>
#include <sofa/component/linearsolver/DiagonalMatrix.h>

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
using std::cerr;
using std::endl;

template<class TMatrix, class TVector>
BlockJacobiPreconditioner<TMatrix,TVector>::BlockJacobiPreconditioner()
    : f_verbose( initData(&f_verbose,false,"verbose","Dump system state at each iteration") )
    , f_graph( initData(&f_graph,"graph","Graph of residuals at each iteration") )
{
    f_graph.setWidget("graph");
    f_graph.setReadOnly(true);
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::solve (Matrix& M, Vector& z, Vector& r)
{
    for (unsigned l=0; l<z.size(); l+=bsize)
    {
        for (unsigned j=0; j<bsize; j++)
        {
            z.set(j+l,0);
            for (unsigned i=0; i<bsize; i++)
            {
                z.add(j+l,M.element(l+i,l+j) * r.element(i+l));
            }
        }
    }

    //M.mult(z,r);
}

template<class TMatrix, class TVector>
void BlockJacobiPreconditioner<TMatrix,TVector>::invert(Matrix& M)
{
    bsize = this->systemMatrix->bandWidth+1;

    for (unsigned l=0; l<M.rowSize(); l+=bsize)
    {
        M.setSubMatrix(l,l,bsize,bsize,M.sub(l,l,bsize,bsize).i());
    }

    //M.i();

    if (f_verbose.getValue()) sout<<M<<sendl;
}


SOFA_DECL_CLASS(BlockJacobiPreconditioner)

int BlockJacobiPreconditionerClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< BlockJacobiPreconditioner<NewMatBandMatrix,NewMatVector> >()
        .add< BlockJacobiPreconditioner<BlockDiagonalMatrix3 ,FullVector<double> > >(true)
        .add< BlockJacobiPreconditioner<BlockDiagonalMatrix6 ,FullVector<double> > >()
        .add< BlockJacobiPreconditioner<BlockDiagonalMatrix9 ,FullVector<double> > >()
        .add< BlockJacobiPreconditioner<BlockDiagonalMatrix12 ,FullVector<double> > >()
        .addAlias("BJCGSolver")
        .addAlias("BJConjugateGradient")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

