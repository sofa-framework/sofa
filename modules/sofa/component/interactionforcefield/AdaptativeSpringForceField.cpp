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
// Author: Rosalie Plantefeve, INRIA, (C) 2013
#define SOFA_COMPONENT_FORCEFIELD_ADAPTATIVESPRINGFORCEFIELD_CPP
#include <sofa/component/interactionforcefield/AdaptativeSpringForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(AdaptativeSpringForceField)

// Register in the Factory
int AdaptativeSpringForceFieldClass = core::RegisterObject("Stiff springs for implicit integration")
#ifndef SOFA_FLOAT
        .add< AdaptativeSpringForceField<Vec3dTypes> >()
        .add< AdaptativeSpringForceField<Vec2dTypes> >()
        .add< AdaptativeSpringForceField<Vec1dTypes> >()
        .add< AdaptativeSpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< AdaptativeSpringForceField<Vec3fTypes> >()
        .add< AdaptativeSpringForceField<Vec2fTypes> >()
        .add< AdaptativeSpringForceField<Vec1fTypes> >()
        .add< AdaptativeSpringForceField<Vec6fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec3dTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec2dTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec1dTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec3fTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec2fTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec1fTypes>;
template class SOFA_DEFORMABLE_API AdaptativeSpringForceField<Vec6fTypes>;
#endif


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

