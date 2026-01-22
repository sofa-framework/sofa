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

#include <sofa/topology/Element.h>

namespace sofa::component::solidmechanics::fem::elastic
{

namespace detail
{

template<class ElementType, class VecCoord, std::size_t... I>
auto extractRefNodesVectorFromGlobalVector(const sofa::topology::Element<ElementType>& element, const VecCoord& vector, std::index_sequence<I...>)
{
    using Coord = typename VecCoord::value_type;
    using CoordRef = std::reference_wrapper<const Coord>;
    return std::array<CoordRef, ElementType::NumberOfNodes>{ std::cref(vector[element[I]])... };
}

template<class ElementType, class VecCoord, std::size_t... I>
auto extractNodesVectorFromGlobalVector(const sofa::topology::Element<ElementType>& element, const VecCoord& vector, std::index_sequence<I...>)
{
    using Coord = typename VecCoord::value_type;
    return std::array<Coord, ElementType::NumberOfNodes>{ vector[element[I]]... };
}

}

template<class ElementType, class VecCoord>
auto extractRefNodesVectorFromGlobalVector(
    const sofa::topology::Element<ElementType>& element, const VecCoord& vector)
{
    return detail::extractRefNodesVectorFromGlobalVector(element, vector, std::make_index_sequence<ElementType::NumberOfNodes>{});
}

template<class ElementType, class VecCoord>
std::array<typename VecCoord::value_type, ElementType::NumberOfNodes>
extractNodesVectorFromGlobalVector(const sofa::topology::Element<ElementType>& element, const VecCoord& vector)
{
    return detail::extractNodesVectorFromGlobalVector(element, vector, std::make_index_sequence<ElementType::NumberOfNodes>{});
}

}
