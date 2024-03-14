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
#include <sofa/linearalgebra/BlockVector.h>

namespace sofa::linearalgebra
{

template<std::size_t N, typename T>
BlockVector<N, T>::BlockVector()
{
}

template<std::size_t N, typename T>
BlockVector<N, T>::BlockVector(Index n)
    : Inherit(n)
{
}

template<std::size_t N, typename T>
BlockVector<N, T>::~BlockVector()
{
}

template<std::size_t N, typename T>
typename BlockVector<N, T>::Block& BlockVector<N, T>::sub(Index i, Index)
{
    return (Block&)*(this->ptr()+i);
}

template<std::size_t N, typename T>
const typename BlockVector<N, T>::Block& BlockVector<N, T>::asub(Index bi, Index) const
{
    return (const Block&)*(this->ptr()+bi*N);
}

template<std::size_t N, typename T>
typename BlockVector<N, T>::Block& BlockVector<N, T>::asub(Index bi, Index)
{
    return (Block&)*(this->ptr()+bi*N);
}

} // namespace sofa::linearalgebra