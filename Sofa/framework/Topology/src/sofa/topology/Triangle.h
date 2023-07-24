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

#include <sofa/topology/Point.h>
#include <sofa/topology/Element.h>

#include <sofa/geometry/Triangle.h>

namespace sofa::topology
{
    using Triangle = sofa::topology::Element<sofa::geometry::Triangle>;

    static constexpr Triangle InvalidTriangle;

    /**
    * @brief	Get the triangle index from one of the point of a triangle in a list and a direction
    * @remark   not really generic
    * @tparam   Real scalar type being used for computations
    * @tparam   Coordinates type representing the coordinates in space (in 3D)
    * @tparam   VectorCoordinates type for the container of coordinates
    * @param	trianglePositions positions of all the points of the triangles set
    * @param	allTriangles informations about the triangle in the topology (a vector of 3 indices, i.e a triangle)
    * @param	shellTriangles triangles around the given point (vector of triangles indices)
    * @param	pointID index of the point (from shellTriangles) from which the algo will try to compute the direction
    * @param	direction direction in which the algo will find a triangle from the pointID
    * @return	index of the triangle in the given direction, if parameters allow it; sofa::InvalidID
    */
    template<typename Coordinates, typename VectorCoordinates, typename Real = SReal>
    static sofa::Index getTriangleIDInDirection(const VectorCoordinates& trianglePositions, const sofa::type::vector<Triangle>& allTriangles,
                                                          const sofa::type::vector<sofa::Index>& shellTriangles, const sofa::Index pointID, 
                                                          const Coordinates& direction)
    {
        sofa::Index indexInTrianglesList = 0;

        for (const auto& tid : shellTriangles)
        {
            const auto& t = allTriangles[tid];
            const auto& c0 = trianglePositions[t[0]];
            const auto& c1 = trianglePositions[t[1]];
            const auto& c2 = trianglePositions[t[2]];

            const sofa::type::Vec<3, Real> p0{ static_cast<Real>(c0[0]), static_cast<Real>(c0[1]) , static_cast<Real>(c0[2]) };
            const sofa::type::Vec<3, Real> p1{ static_cast<Real>(c1[0]), static_cast<Real>(c1[1]) , static_cast<Real>(c1[2]) };
            const sofa::type::Vec<3, Real> p2{ static_cast<Real>(c2[0]), static_cast<Real>(c2[1]) , static_cast<Real>(c2[2]) };

            sofa::type::Vec<3, Real> e1, e2;
            if (t[0] == pointID) { e1 = p1 - p0; e2 = p2 - p0; }
            else if (t[1] == pointID) { e1 = p2 - p1; e2 = p0 - p1; }
            else { e1 = p0 - p2; e2 = p1 - p2; }

            if (const auto v_normal = e2.cross(e1); v_normal.norm2() > static_cast<Real>(0.0))
            {
                const auto v_01 = static_cast<Real>(direction * e1.cross(v_normal));
                const auto v_02 = static_cast<Real>(direction * e2.cross(v_normal));

                const bool is_inside = (v_01 >= static_cast<Real>(0.0)) && (v_02 < static_cast<Real>(0.0));
                if (is_inside) return indexInTrianglesList;
            }
            indexInTrianglesList++;
        }

        return sofa::InvalidID;
    }
}
