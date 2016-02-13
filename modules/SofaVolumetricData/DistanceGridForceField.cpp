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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_DISTANCEGRIDFORCEFIELD_CPP
#include <SofaVolumetricData/DistanceGridForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

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
//.add< DistanceGridForceField<Vec2dTypes> >()
//.add< DistanceGridForceField<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< DistanceGridForceField<Vec3fTypes> >()
//.add< DistanceGridForceField<Vec2fTypes> >()
//.add< DistanceGridForceField<Vec1fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec3dTypes>;
//template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec2dTypes>;
//template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec3fTypes>;
//template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec2fTypes>;
//template class SOFA_VOLUMETRIC_DATA_API DistanceGridForceField<Vec1fTypes>;
#endif

} // namespace forcefield

} // namespace component

} // namespace sofa
