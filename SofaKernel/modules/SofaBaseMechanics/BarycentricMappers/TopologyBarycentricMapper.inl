/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
#ifndef SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_INL
#define SOFA_COMPONENT_MAPPING_TOPOLOGYBARYCENTRICMAPPER_INL

#include "TopologyBarycentricMapper.h"

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _topologybarycentricmapper_
{

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::addPointInLine(const int lineIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::setPointInLine(const int pointIndex, const int lineIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::createPointInLine(const typename Out::Coord& p, int lineIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(lineIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::addPointInTriangle(const int triangleIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::setPointInTriangle(const int pointIndex, const int triangleIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::createPointInTriangle(const typename Out::Coord& p, int triangleIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(triangleIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::addPointInQuad(const int quadIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::setPointInQuad(const int pointIndex, const int quadIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::createPointInQuad(const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(quadIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::addPointInTetra(const int tetraIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::setPointInTetra(const int pointIndex, const int tetraIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::createPointInTetra(const typename Out::Coord& p, int tetraIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(tetraIndex);
    SOFA_UNUSED(points);
    return 0;
}

template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::addPointInCube(const int cubeIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::setPointInCube(const int pointIndex, const int cubeIndex, const SReal* baryCoords)
{
    SOFA_UNUSED(pointIndex);
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(baryCoords);
    return 0;
}
template<class In, class Out>
int TopologyBarycentricMapper<In,Out>::createPointInCube(const typename Out::Coord& p, int cubeIndex, const typename In::VecCoord* points)
{
    SOFA_UNUSED(p);
    SOFA_UNUSED(cubeIndex);
    SOFA_UNUSED(points);
    return 0;
}

}}}}


#endif
