/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/mapping/LineSetSkinningMapping.inl>
#include <sofa/core/componentmodel/behavior/MappedModel.h>
#include <sofa/core/componentmodel/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/componentmodel/behavior/MechanicalMapping.inl>
#include <sofa/core/Mapping.inl>

namespace sofa
{

namespace component
{

namespace mapping
{

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;

SOFA_DECL_CLASS(LineSetSkinningMapping)

// Register in the Factory
int HandMappingClass = core::RegisterObject("skin a model from a set of rigid lines")
#ifndef SOFA_FLOAT
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
// .add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > > >()
//.add< LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
#endif
#endif
//.add< HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > > >()
//.add< HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<Vec3dTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > > >()
//.add< HandMapping< Mapping< MappedModel<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        ;


#ifndef SOFA_FLOAT
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
// template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3fTypes> > >;
template class LineSetSkinningMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3fTypes> > >;
//template class LineSetSkinningMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
#endif
#endif
// Mech -> Mech
//template class HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3dTypes> > >;
//template class HandMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;

// Mech -> Mapped
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3dTypes> > >;
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;

// Mech -> ExtMapped
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3dTypes> > >;
//template class HandMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;

} // namespace mapping

} // namespace component

} // namespace sofa
