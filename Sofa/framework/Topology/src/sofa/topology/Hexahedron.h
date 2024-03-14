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

#include <sofa/geometry/Hexahedron.h>

namespace sofa::topology
{
    using Hexahedron = sofa::topology::Element<sofa::geometry::Hexahedron>;

    template<typename Coordinates, typename VectorCoordinates>
    static constexpr sofa::Index getClosestHexahedronIndex(const VectorCoordinates& hexahedronPositions, const sofa::type::vector<Hexahedron>& hexahedra,
        const Coordinates& pos, type::Vec3& barycentricCoefficients, SReal& distance)
    {
        sofa::Index index = sofa::InvalidID;
        distance = std::numeric_limits<SReal>::max();

        for (sofa::Index c = 0; c < hexahedra.size(); ++c)
        {
            const auto& h = hexahedra[c];
            const auto d = sofa::geometry::Hexahedron::squaredDistanceTo(hexahedronPositions[h[0]], hexahedronPositions[h[1]], hexahedronPositions[h[2]], hexahedronPositions[h[3]],
                hexahedronPositions[h[4]], hexahedronPositions[h[5]], hexahedronPositions[h[6]], hexahedronPositions[h[7]],
                pos);

            if (d < distance)
            {
                distance = d;
                index = c;
            }
        }

        if (index != sofa::InvalidID)
        {
            const auto& h = hexahedra[index];
            barycentricCoefficients = sofa::geometry::Hexahedron::barycentricCoefficients(hexahedronPositions[h[0]], hexahedronPositions[h[1]], hexahedronPositions[h[2]], hexahedronPositions[h[3]],
                hexahedronPositions[h[4]], hexahedronPositions[h[5]], hexahedronPositions[h[6]], hexahedronPositions[h[7]], pos);
        }

        return index;
    }

    static constexpr Hexahedron InvalidHexahedron;

   
}
