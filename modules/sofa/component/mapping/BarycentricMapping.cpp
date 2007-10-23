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

// MeshMapper
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3dTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3fTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3dTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3fTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3dTypes, ExtVec3dTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3fTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3dTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::MeshTopology, Vec3fTypes, ExtVec3dTypes >;

// TriangleSetMapper
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3dTypes>, Vec3dTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3fTypes>, Vec3fTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3dTypes>, Vec3dTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3fTypes>, Vec3fTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3dTypes>, Vec3dTypes, ExtVec3dTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3fTypes>, Vec3fTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3dTypes>, Vec3dTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::TriangleSetTopology<Vec3fTypes>, Vec3fTypes, ExtVec3dTypes >;

// RegularGridMapper
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3dTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3fTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3dTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3fTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3dTypes, ExtVec3dTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3fTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3dTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::RegularGridTopology, Vec3fTypes, ExtVec3dTypes >;

// SparseGridMapper
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3dTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3fTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3dTypes, Vec3fTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3fTypes, Vec3dTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3dTypes, ExtVec3dTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3fTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3dTypes, ExtVec3fTypes >;
template class TopologyBarycentricMapper<topology::SparseGridTopology, Vec3fTypes, ExtVec3dTypes >;

} // namespace mapping

} // namespace component

} // namespace sofa

