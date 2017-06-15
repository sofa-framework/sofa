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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#define SOFA_COMPONENT_FORCEFIELD_STIFFSPRINGFORCEFIELD_CPP
#include <SofaDeformable/StiffSpringForceField.inl>
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


SOFA_DECL_CLASS(StiffSpringForceField)

// Register in the Factory
int StiffSpringForceFieldClass = core::RegisterObject("Stiff springs for implicit integration")
#ifndef SOFA_FLOAT
        .add< StiffSpringForceField<Vec3dTypes> >()
        .add< StiffSpringForceField<Vec2dTypes> >()
        .add< StiffSpringForceField<Vec1dTypes> >()
        .add< StiffSpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< StiffSpringForceField<Vec3fTypes> >()
        .add< StiffSpringForceField<Vec2fTypes> >()
        .add< StiffSpringForceField<Vec1fTypes> >()
        .add< StiffSpringForceField<Vec6fTypes> >()
#endif
        ;
#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec3dTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec2dTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec1dTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec3fTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec2fTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec1fTypes>;
template class SOFA_DEFORMABLE_API StiffSpringForceField<Vec6fTypes>;
#endif


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

