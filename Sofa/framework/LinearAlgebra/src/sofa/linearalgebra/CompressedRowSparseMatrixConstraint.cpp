/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*            (c) 2006-2021 MGH, INRIA, USTL, UJF, CNRS, InSimo                *
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
#define SOFA_LINEARALGEBRA_COMPRESSEDROWSPARSEMATRIXCONSTRAINT_CPP

#include <sofa/linearalgebra/CompressedRowSparseMatrixConstraint.h>
#include <sofa/type/Vec.h>

namespace sofa::linearalgebra
{

template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec1f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec2f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec3f>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec6f>;


template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec1d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec2d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec3d>;
template class SOFA_LINEARALGEBRA_API CompressedRowSparseMatrixConstraint<type::Vec6d>;

} // namespace sofa::linearalgebra
