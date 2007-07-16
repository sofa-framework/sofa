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
#include <sofa/component/mapping/SkinningMapping.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

SOFA_DECL_CLASS(SkinningMapping)

using namespace defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


// Register in the Factory
int SkinningMappingClass = core::RegisterObject("skin a model from a set of rigid dofs")
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< SkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< SkinningMapping< Mapping< MechanicalState<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;
/*
template class SkinningMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SkinningMapping< MechanicalMapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;

template class SkinningMapping< Mapping<MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SkinningMapping< Mapping<MechanicalState<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
template class SkinningMapping< Mapping<MechanicalState<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;

template class SkinningMapping< Mapping<MechanicalState<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SkinningMapping< Mapping<MechanicalState<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
*/


} // namespace mapping

} // namespace component

} // namespace sofa

