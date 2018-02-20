/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_CPP
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>
#include "DistanceGridForceField.inl"

namespace sofa
{

namespace component
{

namespace forcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(DistanceGridForceField)

int DistanceGridForceFieldClass = core::RegisterObject("Force applied by a distancegrid toward the exterior, the interior, or the surface")
#ifndef SOFA_FLOAT
        .add< DistanceGridForceField<Vec3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceGridForceField<Vec3fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_SOFADISTANCEGRID_API DistanceGridForceField<Vec3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_SOFADISTANCEGRID_API DistanceGridForceField<Vec3fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
