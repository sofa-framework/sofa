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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_INL
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTETRAHEDRONSETTOPOLOGY_INL

#include "BarycentricMapperTetrahedronSetTopology.h"

namespace sofa
{

namespace component
{

namespace mapping
{

template <class In, class Out>
BarycentricMapperTetrahedronSetTopology<In,Out>::BarycentricMapperTetrahedronSetTopology(topology::TetrahedronSetTopologyContainer* fromTopology, topology::PointSetTopologyContainer* toTopology)
    : Inherit1(fromTopology, toTopology),
      m_fromContainer(fromTopology),
      m_fromGeomAlgo(NULL)
{}


template <class In, class Out>
int BarycentricMapperTetrahedronSetTopology<In,Out>::addPointInTetra ( const int tetraIndex, const SReal* baryCoords )
{
    helper::vector<MappingData>& vectorData = *(d_map.beginEdit());
    vectorData.resize ( d_map.getValue().size() +1 );
    MappingData& data = *vectorData.rbegin();
    d_map.endEdit();
    data.in_index = tetraIndex;
    data.baryCoords[0] = ( Real ) baryCoords[0];
    data.baryCoords[1] = ( Real ) baryCoords[1];
    data.baryCoords[2] = ( Real ) baryCoords[2];
    return (int)d_map.getValue().size()-1;
}

template <class In, class Out>
helper::vector<Tetrahedron> BarycentricMapperTetrahedronSetTopology<In,Out>::getElements()
{
    return this->m_fromTopology->getTetrahedra();
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperTetrahedronSetTopology<In,Out>::getBaryCoef(const Real* f)
{
    return getBaryCoef(f[0],f[1],f[2]);
}

template <class In, class Out>
helper::vector<SReal> BarycentricMapperTetrahedronSetTopology<In,Out>::getBaryCoef(const Real fx, const Real fy, const Real fz)
{
    helper::vector<SReal> tetrahedronCoef{(1-fx-fy-fz),fx,fy,fz};
    return tetrahedronCoef;
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::computeBase(Mat3x3d& base, const typename In::VecCoord& in, const Tetrahedron& element)
{
    Mat3x3d matrixTranspose;
    base[0] = in[element[1]]-in[element[0]];
    base[1] = in[element[2]]-in[element[0]];
    base[2] = in[element[3]]-in[element[0]];
    matrixTranspose.transpose(base);
    base.invert(matrixTranspose);
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::computeCenter(Vector3& center, const typename In::VecCoord& in, const Tetrahedron& element)
{
    center = ( in[element[0]]+in[element[1]]+in[element[2]]+in[element[3]] ) *0.25;
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::computeDistance(double& d, const Vector3& v)
{
    d = std::max ( std::max ( -v[0],-v[1] ), std::max ( -v[2],v[0]+v[1]+v[2]-1 ) );
}

template <class In, class Out>
void BarycentricMapperTetrahedronSetTopology<In,Out>::addPointInElement(const int elementIndex, const SReal* baryCoords)
{
    addPointInTetra(elementIndex,baryCoords);
}

}}}

#endif
