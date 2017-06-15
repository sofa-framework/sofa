/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.06                  *
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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_BOXSTIFFSPRINGFORCEFIELD_CPP
#include <SofaGeneralObjectInteraction/BoxStiffSpringForceField.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

SOFA_DECL_CLASS(BoxStiffSpringForceField)

int BoxStiffSpringForceFieldClass = core::RegisterObject("Set Spring between the points inside a given box")
#ifndef SOFA_FLOAT
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec3dTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec2dTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec1dTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec3fTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec2fTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec1fTypes> >()
        .add< BoxStiffSpringForceField<sofa::defaulttype::Vec6fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec3dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec2dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec1dTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec3fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec2fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec1fTypes>;
template class SOFA_GENERAL_OBJECT_INTERACTION_API BoxStiffSpringForceField<sofa::defaulttype::Vec6fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

