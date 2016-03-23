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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_PLANEFORCEFIELD_CPP

#include <SofaBoundaryCondition/PlaneForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include <limits>

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(PlaneForceField)

int PlaneForceFieldClass = core::RegisterObject("Repulsion applied by a plane toward the exterior (half-space)")
#ifndef SOFA_FLOAT
.add< PlaneForceField<Vec3dTypes> >()
.add< PlaneForceField<Vec2dTypes> >()
.add< PlaneForceField<Vec1dTypes> >()
.add< PlaneForceField<Vec6dTypes> >()
.add< PlaneForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
.add< PlaneForceField<Vec3fTypes> >() 
.add< PlaneForceField<Vec2fTypes> >()
.add< PlaneForceField<Vec1fTypes> >()
.add< PlaneForceField<Vec6fTypes> >()
.add< PlaneForceField<Rigid3fTypes> >()
#endif
;
#ifndef SOFA_FLOAT
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec3dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec2dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec1dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec6dTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec3fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec2fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec1fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Vec6fTypes>;
template class SOFA_BOUNDARY_CONDITION_API PlaneForceField<Rigid3fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
