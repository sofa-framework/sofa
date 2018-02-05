/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_INTERACTIONELLIPSOIDFORCEFIELD_CPP
#include <SofaGeneralObjectInteraction/InteractionEllipsoidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

//template class InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes>;
//template class InteractionEllipsoidForceField<Vec3dTypes, Vec3dTypes>;
//template class InteractionEllipsoidForceField<Vec3fTypes, Vec3fTypes>;
//template class InteractionEllipsoidForceField<Vec2dTypes, Rigid2dTypes>;
//template class InteractionEllipsoidForceField<Vec2fTypes, Rigid2dTypes>;

SOFA_DECL_CLASS(InteractionEllipsoidForceField)

int EllipsoidForceFieldClass = core::RegisterObject("Repulsion applied by an ellipsoid toward the exterior or the interior")
#ifndef SOFA_FLOAT
        .add< InteractionEllipsoidForceField<Vec3dTypes, Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes> >()
#endif
#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
        .add< InteractionEllipsoidForceField<Vec3dTypes, Rigid3fTypes> >()
        .add< InteractionEllipsoidForceField<Vec3fTypes, Rigid3dTypes> >()
#endif
#endif
//.add< InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes> >()
//.add< InteractionEllipsoidForceField<Vec3dTypes, Vec3dTypes> >()
//.add< InteractionEllipsoidForceField<Vec3fTypes, Vec3fTypes> >()
        ;

#ifndef SOFA_FLOAT
template class InteractionEllipsoidForceField<Vec3dTypes, Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class InteractionEllipsoidForceField<Vec3fTypes, Rigid3fTypes>;
#endif

#ifndef SOFA_FLOAT
#ifndef SOFA_DOUBLE
template class InteractionEllipsoidForceField<Vec3dTypes, Rigid3fTypes>;
template class InteractionEllipsoidForceField<Vec3fTypes, Rigid3dTypes>;
#endif
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
