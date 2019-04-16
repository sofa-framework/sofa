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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERQUADSETTOPOLOGY_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERQUADSETTOPOLOGY_INL


#include "BarycentricMapperQuadSetTopology.h"

namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
BarycentricMapperQuadSetTopology<In,Out>::BarycentricMapperQuadSetTopology(topology::QuadSetTopologyContainer* fromTopology,
                                                                           topology::PointSetTopologyContainer* toTopology)
    : Inherit1(fromTopology, toTopology),
      m_fromContainer(fromTopology),
      m_fromGeomAlgo(NULL)
{}

template <class In, class Out>
BarycentricMapperQuadSetTopology<In,Out>::~BarycentricMapperQuadSetTopology()
{}


template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::addPointInQuad ( const int quadIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = quadIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out>
int BarycentricMapperQuadSetTopology<In,Out>::createPointInQuad ( const typename Out::Coord& p, int quadIndex, const typename In::VecCoord* points )
{
    SReal baryCoords[2];
    const Quad& elem = this->m_fromTopology->getQuad ( quadIndex );
    const typename In::Coord p0 = ( *points ) [elem[0]];
    const typename In::Coord pA = ( *points ) [elem[1]] - p0;
    const typename In::Coord pB = ( *points ) [elem[3]] - p0;
    typename In::Coord pos = Out::getCPos(p) - p0;
    sofa::defaulttype::Mat<3,3,typename In::Real> m,mt,base;
    m[0] = pA;
    m[1] = pB;
    m[2] = cross ( pA, pB );
    mt.transpose ( m );
    base.invert ( mt );
    const typename In::Coord base0 = base[0];
    const typename In::Coord base1 = base[1];
    baryCoords[0] = base0 * pos;
    baryCoords[1] = base1 * pos;
    return this->addPointInQuad ( quadIndex, baryCoords );
}

template <class In, class Out>
helper::vector<Quad> BarycentricMapperQuadSetTopology<In,Out>::getElements()
{
    return this->m_fromTopology->getQuads();
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperQuadSetTopology<In,Out>::getBaryCoef(const Real* f)
{
    return getBaryCoef(f[0],f[1]);
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperQuadSetTopology<In,Out>::getBaryCoef(const Real fx, const Real fy)
{
    helper::vector<SReal> quadCoef{(1-fx)*(1-fy),
                (fx)*(1-fy),
                (1-fx)*(fy),
                (fx)*(fy)};
    return quadCoef;
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Quad& element)
{
    Mat3x3d matrixTranspose;
    base[0] = in[element[1]]-in[element[0]];
    base[1] = in[element[3]]-in[element[0]];
    base[2] = cross(base[0],base[1]);
    matrixTranspose.transpose(base);
    base.invert(matrixTranspose);
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::computeCenter(Vector3& center, const typename In::VecCoord& in, const Quad& element)
{
    center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]] ) *0.25;
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::computeDistance(double& d, const Vector3& v)
{
    d = std::max ( std::max ( -v[0],-v[1] ),std::max ( std::max ( v[1]-1,v[0]-1 ),std::max ( v[2]-0.01,-v[2]-0.01 ) ) );
}

template <class In, class Out>
void BarycentricMapperQuadSetTopology<In,Out>::addPointInElement(const int elementIndex, const SReal* baryCoords)
{
    addPointInQuad(elementIndex,baryCoords);
}

}}}

#endif
