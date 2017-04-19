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
#define SOFA_COMPONENT_MAPPING_BARYCENTRICMAPPING_CPP
#include <SofaBaseMechanics/BarycentricMapping.inl>

#include <sofa/defaulttype/VecTypes.h>

#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(BarycentricMapping)

// Register in the Factory
int BarycentricMappingClass = core::RegisterObject("Mapping using barycentric coordinates of the child with respect to cells of its parent")
#ifndef SOFA_FLOAT
        .add< BarycentricMapping< Vec3dTypes, Vec3dTypes > >()
        .add< BarycentricMapping< Vec3dTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_DOUBLE
        .add< BarycentricMapping< Vec3fTypes, Vec3fTypes > >()
        .add< BarycentricMapping< Vec3fTypes, ExtVec3fTypes > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< BarycentricMapping< Vec3fTypes, Vec3dTypes > >()
        .add< BarycentricMapping< Vec3dTypes, Vec3fTypes > >()
#endif
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, ExtVec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, ExtVec3fTypes >;
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapping< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapper< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API TopologyBarycentricMapper< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperMeshTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3dTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3fTypes >;
template class SOFA_BASE_MECHANICS_API BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3dTypes >;
#endif
#endif

// Quick and dirty way to make sure all mapper metaclasses are instantiated before multiple threads can be used
static const sofa::core::objectmodel::BaseClass* classes[] = {
#ifndef SOFA_FLOAT
    BarycentricMapper< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapper< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3dTypes, Vec3dTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3dTypes, ExtVec3fTypes >::GetClass(),
#endif
#ifndef SOFA_DOUBLE
    BarycentricMapper< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapper< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3fTypes, Vec3fTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3fTypes, ExtVec3fTypes >::GetClass(),
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
    BarycentricMapper< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapper< Vec3fTypes, Vec3dTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3dTypes, Vec3fTypes >::GetClass(),
    TopologyBarycentricMapper< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperRegularGridTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperSparseGridTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperMeshTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperEdgeSetTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperTriangleSetTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperQuadSetTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperTetrahedronSetTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3dTypes, Vec3fTypes >::GetClass(),
    BarycentricMapperHexahedronSetTopology< Vec3fTypes, Vec3dTypes >::GetClass(),
#endif
#endif
    NULL
};

} // namespace mapping

} // namespace component

} // namespace sofa
