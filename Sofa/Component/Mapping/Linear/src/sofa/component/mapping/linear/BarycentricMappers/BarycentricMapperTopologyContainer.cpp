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
#include <sofa/component/mapping/linear/BarycentricMappers/BarycentricMapperTopologyContainer.inl>

namespace sofa::component::mapping::linear::_barycentricmappertopologycontainer_
{

using namespace sofa::defaulttype;

template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData1D, Edge>;
template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData2D, Triangle>;
template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData2D, Quad>;
template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData3D, Tetrahedron>;
template class SOFA_COMPONENT_MAPPING_LINEAR_API BarycentricMapperTopologyContainer< Vec3Types, Vec3Types , typename BarycentricMapper<Vec3Types, Vec3Types>::MappingData3D, Hexahedron>;

} // namespace sofa::component::mapping::linear::_barycentricmappertopologycontainer_
