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

#include <sofa/topology/geometry/Point.h>
#include <sofa/type/stdtype/fixed_array.h>

namespace sofa::topology::geometry
{
    using EdgeID = Index;

    class Edge : public sofa::type::stdtype::fixed_array<PointID, 2>
    {
    public:
        Edge() : sofa::type::stdtype::fixed_array<PointID, 2>(InvalidID, InvalidID) {}
        Edge(PointID a, PointID b) : sofa::type::stdtype::fixed_array<PointID, 2>(a, b) {}
    };

    inline static const Edge InvalidEdge;
}
