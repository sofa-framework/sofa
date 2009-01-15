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
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/linearsolver/NewMatMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>
#include <iostream>
#include "sofa/helper/system/thread/CTime.h"

namespace sofa
{

namespace component
{

namespace linearsolver
{

using namespace sofa::defaulttype;
using namespace sofa::core::componentmodel::behavior;
using namespace sofa::simulation;

SOFA_DECL_CLASS(CGLinearSolver)

int CGLinearSolverClass = core::RegisterObject("Linear system solver using the conjugate gradient iterative algorithm")
        .add< CGLinearSolver<GraphScatteredMatrix,GraphScatteredVector> >(true)
        .add< CGLinearSolver<NewMatMatrix,NewMatVector> >()
        .add< CGLinearSolver<NewMatSymmetricMatrix,NewMatVector> >()
        .add< CGLinearSolver<NewMatBandMatrix,NewMatVector> >()
        .add< CGLinearSolver<NewMatSymmetricBandMatrix,NewMatVector> >()
        .add< CGLinearSolver< FullMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< SparseMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<double>, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<float>, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<2,2,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<3,3,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<4,4,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<6,6,float> >, FullVector<float> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,double> >, FullVector<double> > >()
        .add< CGLinearSolver< CompressedRowSparseMatrix<Mat<8,8,float> >, FullVector<float> > >()
        .addAlias("CGSolver")
        .addAlias("ConjugateGradient")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

