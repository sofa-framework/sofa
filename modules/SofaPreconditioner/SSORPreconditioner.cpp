/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <SofaPreconditioner/SSORPreconditioner.inl>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>


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

SOFA_DECL_CLASS(SSORPreconditioner)

int SSORPreconditionerClass = core::RegisterObject("Linear system solver / preconditioner based on Symmetric Successive Over-Relaxation (SSOR). If the matrix is decomposed as $A = D + L + L^T$, this solver computes $(1/(2-w))(D/w+L)(D/w)^{-1}(D/w+L)^T x = b, or $(D+L)D^{-1}(D+L)^T x = b$ if $w=1$.")
//.add< SSORPreconditioner<GraphScatteredMatrix,GraphScatteredVector> >(true)
// .add< SSORPreconditioner< SparseMatrix<double>, FullVector<double> > >()
        .add< SSORPreconditioner< CompressedRowSparseMatrix<double>, FullVector<double> > >(true)
        .add< SSORPreconditioner< CompressedRowSparseMatrix< defaulttype::Mat<3,3,double> >, FullVector<double> > >()
//.add< SSORPreconditioner<NewMatBandMatrix,NewMatVector> >(true)
//.add< SSORPreconditioner<NewMatMatrix,NewMatVector> >()
// .add< SSORPreconditioner<NewMatSymmetricMatrix,NewMatVector> >()
//.add< SSORPreconditioner<NewMatSymmetricBandMatrix,NewMatVector> >()
// .add< SSORPreconditioner< FullMatrix<double>, FullVector<double> > >()
        .addAlias("SSORLinearSolver")
        .addAlias("SSORSolver")
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

