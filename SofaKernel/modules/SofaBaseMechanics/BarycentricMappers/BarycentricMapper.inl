/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPER_INL

#include "BarycentricMapper.h"

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _barycentricmapper_
{

using sofa::component::linearsolver::CompressedRowSparseMatrix;

template<class In, class Out>
void BarycentricMapper<In,Out>::addMatrixContrib(CompressedRowSparseMatrix<MBloc>* m, int row, int col, Real value)
{
    MBloc* b = m->wbloc(row, col, true); // get write access to a matrix bloc, creating it if not found
    for (int i=0; i < ((int)NIn < (int)NOut ? (int)NIn : (int)NOut); ++i)
        (*b)[i][i] += value;
}


template<class In, class Out>
const sofa::defaulttype::BaseMatrix* BarycentricMapper<In,Out>::getJ(int outSize, int inSize)
{
    SOFA_UNUSED(outSize);
    SOFA_UNUSED(inSize);
    dmsg_error() << " getJ() NOT IMPLEMENTED BY " << sofa::core::objectmodel::BaseClass::decodeClassName(typeid(*this)) ;
    return nullptr;
}

template<class In, class Out>
void BarycentricMapper<In,Out>::applyOnePoint( const unsigned int& hexaId, typename Out::VecCoord& out, const typename In::VecCoord& in)
{
    SOFA_UNUSED(hexaId);
    SOFA_UNUSED(out);
    SOFA_UNUSED(in);
}

}}}}

#endif
