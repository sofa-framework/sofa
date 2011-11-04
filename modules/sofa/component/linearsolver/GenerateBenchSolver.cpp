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
// Author: Hadrien Courtecuisse
//
// Copyright: See COPYING file that comes with this distribution
#include "GenerateBenchSolver.inl"
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.inl>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace linearsolver
{


SOFA_DECL_CLASS(GenerateBenchSolver)

int GenerateBenchSolverClass = core::RegisterObject("Generate a bench for sofaCUDA_BENCH")
        .add< GenerateBenchSolver< sofa::component::linearsolver::CompressedRowSparseMatrix<double>, FullVector<double> > >()
        .add< GenerateBenchSolver< sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,double> >, FullVector<double> > >(true)
        .add< GenerateBenchSolver< sofa::component::linearsolver::CompressedRowSparseMatrix<float>, FullVector<float> > >()
        .add< GenerateBenchSolver< sofa::component::linearsolver::CompressedRowSparseMatrix<defaulttype::Mat<3,3,float> >, FullVector<float> > >()
        ;

} // namespace linearsolver

} // namespace component

} // namespace sofa

