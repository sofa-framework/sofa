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
#include <sofa/component/linearsystem/CompositeLinearSystem.inl>
#include <sofa/core/ObjectFactory.h>

#include <sofa/linearalgebra/FullMatrix.h>
#include <sofa/linearalgebra/CompressedRowSparseMatrix.h>
#include <sofa/linearalgebra/SparseMatrix.h>
#include <sofa/linearalgebra/DiagonalMatrix.h>
#include <sofa/linearalgebra/RotationMatrix.h>
#include <sofa/linearalgebra/BlockDiagonalMatrix.h>

namespace sofa::component::linearsystem
{

using namespace sofa::linearalgebra;

template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< FullMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< SparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< DiagonalMatrix<SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< BlockDiagonalMatrix<3,SReal>, FullVector<SReal> >;
template class SOFA_COMPONENT_LINEARSYSTEM_API CompositeLinearSystem< RotationMatrix<SReal>, FullVector<SReal> >;

int CompositeLinearSystemClass = sofa::core::RegisterObject("Component acting like a linear system, but delegates the linear system functionalities to a list of real linear systems")
    .add<CompositeLinearSystem< FullMatrix<SReal>, FullVector<SReal> > >()
    .add<CompositeLinearSystem< SparseMatrix<SReal>, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<SReal>, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<2,2,SReal> >, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<3,3,SReal> >, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<4,4,SReal> >, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<6,6,SReal> >, FullVector<SReal> > >()
    .add<CompositeLinearSystem< CompressedRowSparseMatrix<type::Mat<8,8,SReal> >, FullVector<SReal> > >()
    .add<CompositeLinearSystem< DiagonalMatrix<SReal>, FullVector<SReal> > >()
    .add<CompositeLinearSystem< BlockDiagonalMatrix<3,SReal>, FullVector<SReal> > >()
    .add<CompositeLinearSystem< RotationMatrix<SReal>, FullVector<SReal> > >()
;
}

