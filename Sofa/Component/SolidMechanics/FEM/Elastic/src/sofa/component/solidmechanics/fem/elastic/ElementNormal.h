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

#include <sofa/type/Mat.h>
#include <sofa/type/Vec.h>

namespace sofa::component::solidmechanics::fem::elastic
{

/**
 * @brief Unit normal of a codimension-1 element, from the jacobian of its mapping.
 *
 * Defined only where the element spans one dimension less than the space it lives in: a surface
 * element in 3D, an edge in 2D. Its orientation follows the node ordering of the element.
 *
 * @param jacobian dx/dq of the reference-to-physical mapping at the point of interest.
 */
template <sofa::Size spatial_dimensions, sofa::Size TopologicalDimension, class Real>
sofa::type::Vec<spatial_dimensions, Real> elementNormal(
    const sofa::type::Mat<spatial_dimensions, TopologicalDimension, Real>& jacobian)
{
    static_assert(TopologicalDimension + 1 == spatial_dimensions,
        "A normal is only defined for an element of codimension 1.");

    if constexpr (spatial_dimensions == 3)
    {
        return jacobian.col(0).cross(jacobian.col(1)).normalized();
    }
    else
    {
        const sofa::type::Vec<2, Real> tangent = jacobian.col(0);
        return sofa::type::Vec<2, Real>(tangent[1], -tangent[0]).normalized();
    }
}

}  // namespace sofa::component::solidmechanics::fem::elastic
