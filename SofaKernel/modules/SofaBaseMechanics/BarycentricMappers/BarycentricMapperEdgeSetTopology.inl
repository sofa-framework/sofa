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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPEREDGESETTOPOLOGY_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPEREDGESETTOPOLOGY_INL

#include "BarycentricMapperEdgeSetTopology.h"

namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::addPointInLine ( const int edgeIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    d_map.endEdit();
    MappingData& data = *vectorData.rbegin();
    data.in_index = edgeIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperEdgeSetTopology<In,Out>::createPointInLine ( const typename Out::Coord& p, int edgeIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[1];
    const Edge& elem = this->m_fromTopology->getEdge ( edgeIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    baryCoords[0] = dot ( pA,pos ) /dot ( pA,pA );
    return this->addPointInLine ( edgeIndex, baryCoords );
}


template <class In, class Out>
helper::vector<Edge> BarycentricMapperEdgeSetTopology<In,Out>::getElements()
{
    return this->m_fromTopology->getEdges();
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperEdgeSetTopology<In,Out>::getBaryCoef(const Real* f)
{
    return getBaryCoef(f[0]);
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperEdgeSetTopology<In,Out>::getBaryCoef(const Real fx)
{
    helper::vector<SReal> edgeCoef{1-fx,fx};
    return edgeCoef;
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Edge& element)
{
    //Not implemented for Edge
    SOFA_UNUSED(base);
    SOFA_UNUSED(in);
    SOFA_UNUSED(element);
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::computeCenter(Vector3& center, const typename In::VecCoord& in, const Edge& element)
{
    center = (in[element[0]]+in[element[1]])*0.5;
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::computeDistance(double& d, const Vector3& v)
{
    //Not implemented for Edge
    SOFA_UNUSED(d);
    SOFA_UNUSED(v);
}

template <class In, class Out>
void BarycentricMapperEdgeSetTopology<In,Out>::addPointInElement(const int elementIndex, const SReal* baryCoords)
{
    addPointInLine(elementIndex,baryCoords);
}

}}}

#endif
