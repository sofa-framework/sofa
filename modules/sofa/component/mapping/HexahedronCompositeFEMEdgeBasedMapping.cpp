/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2016 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <sofa/component/mapping/HexahedronCompositeFEMEdgeBasedMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/MappedModel.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(HexahedronCompositeFEMEdgeBasedMapping)

using namespace defaulttype;
using namespace core;
using namespace core::behavior;


// Register in the Factory
int HexahedronCompositeFEMEdgeBasedMappingClass = core::RegisterObject("Set the point to the center of mass of the DOFs it is attached to")
#ifndef SOFA_FLOAT
        .add< HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
#endif
//
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
#endif
#endif
        ;



#ifndef SOFA_FLOAT
template class HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
//
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class HexahedronCompositeFEMEdgeBasedMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
;



} // namespace mapping

} // namespace component

} // namespace sofa

