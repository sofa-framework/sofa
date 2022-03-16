/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_SSORPRECONDITIONER_CPP
#include <sofa/component/linearsolver/preconditioner/SSORPreconditioner.inl>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::linearsolver::preconditioner
{

using namespace sofa::defaulttype;
using namespace sofa::core::objectmodel;
using namespace sofa::linearalgebra;

int SSORPreconditionerClass = core::RegisterObject("Linear system solver / preconditioner based on Symmetric Successive Over-Relaxation (SSOR). If the matrix is decomposed as $A = D + L + L^T$, this solver computes $(1/(2-w))(D/w+L)(D/w)^{-1}(D/w+L)^T x = b, or $(D+L)D^{-1}(D+L)^T x = b$ if $w=1$.")
        .add< SSORPreconditioner< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >(true)
        .add< SSORPreconditioner< CompressedRowSparseMatrix< type::Mat<3,3,SReal> >, FullVector<SReal> > >()
        .addAlias("SSORLinearSolver")
        .addAlias("SSORSolver")
        ;

template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSOLVER_PRECONDITIONER_API SSORPreconditioner< CompressedRowSparseMatrix< type::Mat<3, 3, SReal> >, FullVector<SReal> >;

} // namespace sofa::component::linearsolver::preconditioner
