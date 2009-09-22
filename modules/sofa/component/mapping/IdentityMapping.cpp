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
#define SOFA_COMPONENT_MAPPING_IDENTITYMAPPING_CPP
#include <sofa/component/mapping/IdentityMapping.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
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

using namespace sofa::defaulttype;
using namespace core;
using namespace core::componentmodel::behavior;


SOFA_DECL_CLASS(IdentityMapping)

// Register in the Factory
int IdentityMappingClass = core::RegisterObject("Special case of mapping where the child points are the same as the parent points")
#ifndef SOFA_FLOAT
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Rigid2dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > > >()
// .add< IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2dTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Rigid2fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2fTypes> > > >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Rigid2fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Rigid2dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2dTypes> > > >()
#endif
#endif

// Rigid -> Vec
#ifndef SOFA_FLOAT
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Vec2dTypes> > > >()
// .add< IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
#ifndef SOFA_DOUBLE
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > > >()
        .add< IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Vec2fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > > >()
        .add< IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > > >()
#endif
        ;


#ifndef SOFA_FLOAT
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Rigid2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Vec3dTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<ExtVec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2dTypes>, MechanicalState<Vec2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Vec2dTypes> > >;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3fTypes> > >;
// template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<ExtVec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Rigid2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<ExtVec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid2fTypes>, MechanicalState<Vec2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Vec2fTypes> > >;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3dTypes>, MechanicalState<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec3fTypes>, MechanicalState<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2dTypes>, MechanicalState<Vec2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec2fTypes>, MechanicalState<Vec2dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1dTypes>, MechanicalState<Vec1fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec1fTypes>, MechanicalState<Vec1dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6dTypes>, MechanicalState<Vec6fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Vec6fTypes>, MechanicalState<Vec6dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3dTypes>, MappedModel<Vec3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Vec3fTypes>, MappedModel<Vec3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3fTypes>, MechanicalState<Rigid3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< MechanicalMapping< MechanicalState<Rigid3dTypes>, MechanicalState<Rigid3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3fTypes>, MappedModel<Rigid3dTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid3dTypes>, MappedModel<Rigid3fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2dTypes>, MappedModel<Rigid2fTypes> > >;
template class SOFA_COMPONENT_MAPPING_API IdentityMapping< Mapping< State<Rigid2fTypes>, MappedModel<Rigid2dTypes> > >;
#endif
#endif


} // namespace mapping

} // namespace component

} // namespace sofa

