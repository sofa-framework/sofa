/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_MAPPING_BEZIER2MESHMECHANICALMAPPING_CPP
#include "Bezier2MeshMechanicalMapping.inl"


#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(Bezier2MeshMechanicalMapping)

int Bezier2MeshMechanicalMappingClass = core::RegisterObject("Mechanical mapping between a Bezier triangle or Bezier tetrahedra with a tesselated triangle mesh or tesselated tetrahedron mesh")
#ifndef SOFA_FLOAT
        .add< Bezier2MeshMechanicalMapping< Vec3dTypes, Vec3dTypes > >()
        .add< Bezier2MeshMechanicalMapping< Vec3dTypes, ExtVec3dTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< Bezier2MeshMechanicalMapping< Vec3fTypes, Vec3fTypes > >()
        .add< Bezier2MeshMechanicalMapping< Vec3fTypes, ExtVec3fTypes > >()
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< Bezier2MeshMechanicalMapping< Vec3fTypes, Vec3dTypes > >()
        .add< Bezier2MeshMechanicalMapping< Vec3dTypes, Vec3fTypes > >()
		.add< Bezier2MeshMechanicalMapping< Vec3dTypes, ExtVec3fTypes > >()
		.add< Bezier2MeshMechanicalMapping< Vec3fTypes, ExtVec3dTypes > >()
#endif
#endif
//.addAlias("SimpleTesselatedTetraMechanicalMapping")
        ;


#ifndef SOFA_FLOAT
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3dTypes, ExtVec3dTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3fTypes, ExtVec3fTypes >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3fTypes, Vec3dTypes >;
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3fTypes, ExtVec3dTypes >;
template class SOFA_TOPOLOGY_MAPPING_API Bezier2MeshMechanicalMapping< Vec3dTypes, ExtVec3fTypes >;
#endif
#endif

} // namespace mapping

} // namespace component

} // namespace sofa
