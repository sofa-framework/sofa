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
#define SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP

#include <SofaBoundaryCondition/EllipsoidForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(EllipsoidForceField)

int EllipsoidForceFieldClass = core::RegisterObject("Repulsion applied by an ellipsoid toward the exterior or the interior")

#ifndef SOFA_FLOAT
        .add< EllipsoidForceField<Vec3dTypes> >()
        .add< EllipsoidForceField<Vec2dTypes> >()
        .add< EllipsoidForceField<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< EllipsoidForceField<Vec3fTypes> >()
        .add< EllipsoidForceField<Vec2fTypes> >()
        .add< EllipsoidForceField<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API EllipsoidForceField<Vec1fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
