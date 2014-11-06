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
#include <SofaDeformable/QuadBendingSprings.inl>
#include <SofaDeformable/StiffSpringForceField.inl>
#include <sofa/SofaCommon.h>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(QuadBendingSprings)

// Register in the Factory
int QuadBendingSpringsClass = core::RegisterObject("Springs added to a quad mesh to prevent bending")
#ifndef SOFA_FLOAT
        .add< QuadBendingSprings<Vec3dTypes> >()
        .add< QuadBendingSprings<Vec2dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< QuadBendingSprings<Vec3fTypes> >()
        .add< QuadBendingSprings<Vec2fTypes> >()
#endif
        ;


#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API QuadBendingSprings<Vec3dTypes>;
template class SOFA_DEFORMABLE_API QuadBendingSprings<Vec2dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API QuadBendingSprings<Vec3fTypes>;
template class SOFA_DEFORMABLE_API QuadBendingSprings<Vec2fTypes>;
#endif


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

