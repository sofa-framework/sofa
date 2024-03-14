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
#include <sofa/component/mapping/linear/BarycentricMappers/TopologyBarycentricMapper.h>

namespace sofa::component::mapping::linear::_topologybarycentricmapper_
{

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::addPointInLine(const Index lineIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::setPointInLine(const Index pointIndex, const Index lineIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::createPointInLine(const typename Out::Coord& p, Index lineIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::addPointInTriangle(const Index triangleIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::setPointInTriangle(const Index pointIndex, const Index triangleIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::createPointInTriangle(const typename Out::Coord& p, Index triangleIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::addPointInQuad(const Index quadIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::setPointInQuad(const Index pointIndex, const Index quadIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::createPointInQuad(const typename Out::Coord& p, Index quadIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::addPointInTetra(const Index tetraIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::setPointInTetra(const Index pointIndex, const Index tetraIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::createPointInTetra(const typename Out::Coord& p, Index tetraIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::addPointInCube(const Index cubeIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::setPointInCube(const Index pointIndex, const Index cubeIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
typename TopologyBarycentricMapper<In, Out>::Index 
TopologyBarycentricMapper<In,Out>::createPointInCube(const typename Out::Coord& p, Index cubeIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(points);
    return 0;
}

} // namespace sofa::component::mapping::linear::_topologybarycentricmapper_

