/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include <sofa/component/interactionforcefield/BoxStiffSpringForceField.inl>
#include <sofa/component/interactionforcefield/StiffSpringForceField.inl>
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
        .add< BoxStiffSpringForceField<Vec3dTypes> >()
        .add< BoxStiffSpringForceField<Vec2dTypes> >()
        .add< BoxStiffSpringForceField<Vec1dTypes> >()
        .add< BoxStiffSpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< BoxStiffSpringForceField<Vec3fTypes> >()
        .add< BoxStiffSpringForceField<Vec2fTypes> >()
        .add< BoxStiffSpringForceField<Vec1fTypes> >()
        .add< BoxStiffSpringForceField<Vec6fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec3dTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec2dTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec1dTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec3fTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec2fTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec1fTypes>;
template class SOFA_OBJECT_INTERACTION_API BoxStiffSpringForceField<Vec6fTypes>;
#endif
} // namespace interactionforcefield

} // namespace component

} // namespace sofa

