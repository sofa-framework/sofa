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
#pragma once
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapper.h>

namespace sofa::component::mapping::linear::_barycentricmapper_
{

using sofa::linearalgebra::CompressedRowSparseMatrix;

template<class In, class Out>
void BarycentricMapper<In,Out>::addMatrixContrib(CompressedRowSparseMatrix<MBloc>* m, sofa::Index row, sofa::Index col, Real value)
{
    MBloc* b = m->wblock(row, col, true); // get write access to a matrix block, creating it if not found
    for (sofa::Index i=0; i < (NIn < NOut ? NIn : NOut); ++i)
        (*b)[i][i] += value;
}


template<class In, class Out>
const sofa::linearalgebra::BaseMatrix* BarycentricMapper<In,Out>::getJ(int outSize, int inSize)
{
    SOFA_UNUSED(outSize);
    SOFA_UNUSED(inSize);
    dmsg_error() << " getJ() NOT IMPLEMENTED BY " << sofa::helper::NameDecoder::decodeClassName(typeid(*this)) ;
    return nullptr;
}

template<class In, class Out>
void BarycentricMapper<In,Out>::applyOnePoint( const Index& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    SOFA_UNUSED(hexaId);
    SOFA_UNUSED(out);
    SOFA_UNUSED(in);
}

} // namespace sofa::component::mapping::linear::_barycentricmapper_
