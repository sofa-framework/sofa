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
#ifndef SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_CPP
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPERTOPOLOGYCONTAINER_CPP

#include "BarycentricMapperTopologyContainer.inl"

namespace sofa
{

namespace component
{

namespace mapping
{

namespace _barycentricmappertopologycontainer_
{

using namespace sofa::defaulttype;

#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData1D, Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData1D, Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData2D, Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData2D, Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData2D, Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData2D, Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData3D, Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData3D, Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3dTypes , typename BarycentricMapper<Vec3dTypes, Vec3dTypes>::MappingData3D, Hexahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3dTypes, ExtVec3fTypes>::MappingData3D, Hexahedron>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData1D, Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData1D, Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData2D, Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData2D, Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData2D, Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData2D, Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData3D, Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData3D, Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3fTypes , typename BarycentricMapper<Vec3fTypes, Vec3fTypes>::MappingData3D, Hexahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, ExtVec3fTypes , typename BarycentricMapper<Vec3fTypes, ExtVec3fTypes>::MappingData3D, Hexahedron>;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData1D,Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData1D,Edge>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData2D,Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData2D,Triangle>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData2D,Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData2D,Quad>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData3D,Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData3D,Tetrahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3dTypes, Vec3fTypes , typename BarycentricMapper<Vec3dTypes, Vec3fTypes>::MappingData3D,Hexahedron>;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTopologyContainer< Vec3fTypes, Vec3dTypes , typename BarycentricMapper<Vec3fTypes, Vec3dTypes>::MappingData3D,Hexahedron>;
#endif
#endif

}}}}


#endif
