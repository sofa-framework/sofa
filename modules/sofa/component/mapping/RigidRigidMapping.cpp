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
#include <sofa/component/mapping/RigidRigidMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(RigidRigidMapping)

using namespace defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


// Register in the Factory
int RigidRigidMappingClass = core::RegisterObject("Set the positions and velocities of points attached to a rigid parent")
        .add< RigidRigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< RigidRigidMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< RigidRigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< RigidRigidMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > > >()
        .add< RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > > >()
        .add< RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > > >()
        .add< RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > > >()
        ;

template class RigidRigidMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > >;
template class RigidRigidMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa

