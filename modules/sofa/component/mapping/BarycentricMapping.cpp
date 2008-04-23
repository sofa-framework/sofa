/*******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 1       *
*                (c) 2006-2007 MGH, INRIA, USTL, UJF, CNRS                     *
*                                                                              *
* This library is free software; you can redistribute it and/or modify it      *
* under the terms of the GNU Lesser General Public License as published by the *
* Free Software Foundation; either version 2.1 of the License, or (at your     *
* option) any later version.                                                   *
*                                                                              *
* This library is distributed in the hope that it will be useful, but WITHOUT  *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License  *
* for more details.                                                            *
*                                                                              *
* You should have received a copy of the GNU Lesser General Public License     *
* along with this library; if not, write to the Free Software Foundation,      *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.           *
*                                                                              *
* Contact information: contact@sofa-framework.org                              *
*                                                                              *
* Authors: J. Allard, P-J. Bensoussan, S. Cotin, C. Duriez, H. Delingette,     *
* F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza, M. Nesme, P. Neumann,        *
* and F. Poyer                                                                 *
*******************************************************************************/
#include <sofa/component/mapping/BarycentricMapping.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.h>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(BarycentricMapping)

// Register in the Factory
int BarycentricMappingClass = core::RegisterObject("Mapping using barycentric coordinates of the child with respect to cells of its parent")
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;

// Mech -> Mech
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class BarycentricMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;

// Mech -> Mapped
template class BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;

// Mech -> ExtMapped
template class BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class BarycentricMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;

// RegularGridMapper
template class BarycentricMapperRegularGridTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperRegularGridTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperRegularGridTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperRegularGridTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperRegularGridTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperRegularGridTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperRegularGridTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperRegularGridTopology<Vec3fTypes, ExtVec3dTypes >;

// SparseGridMapper
template class BarycentricMapperSparseGridTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperSparseGridTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperSparseGridTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperSparseGridTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperSparseGridTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperSparseGridTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperSparseGridTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperSparseGridTopology<Vec3fTypes, ExtVec3dTypes >;

// MeshMapper
template class BarycentricMapperMeshTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperMeshTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperMeshTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperMeshTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperMeshTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperMeshTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperMeshTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperMeshTopology<Vec3fTypes, ExtVec3dTypes >;

// EdgeSetTopologyMapper
template class BarycentricMapperEdgeSetTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperEdgeSetTopology<Vec3fTypes, ExtVec3dTypes >;

// TriangleSetTopologyMapper
template class BarycentricMapperTriangleSetTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperTriangleSetTopology<Vec3fTypes, ExtVec3dTypes >;

// QuadSetTopologyMapper
template class BarycentricMapperQuadSetTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperQuadSetTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperQuadSetTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperQuadSetTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperQuadSetTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperQuadSetTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperQuadSetTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperQuadSetTopology<Vec3fTypes, ExtVec3dTypes >;

// TetrahedronSetTopologyMapper
template class BarycentricMapperTetrahedronSetTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperTetrahedronSetTopology<Vec3fTypes, ExtVec3dTypes >;

// HexahedronSetTopologyMapper
template class BarycentricMapperHexahedronSetTopology<Vec3dTypes, Vec3dTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3fTypes, Vec3fTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3dTypes, Vec3fTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3fTypes, Vec3dTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3dTypes, ExtVec3dTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3fTypes, ExtVec3fTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3dTypes, ExtVec3fTypes >;
template class BarycentricMapperHexahedronSetTopology<Vec3fTypes, ExtVec3dTypes >;

} // namespace mapping

} // namespace component

} // namespace sofa

