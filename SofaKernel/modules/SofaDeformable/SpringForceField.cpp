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
// Author: François Faure, INRIA-UJF, (C) 2006
//
// Copyright: See COPYING file that comes with this distribution
#define SOFA_COMPONENT_FORCEFIELD_SPRINGFORCEFIELD_CPP
#include <SofaDeformable/SpringForceField.inl>
#include <sofa/defaulttype/Vec3Types.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

SOFA_DECL_CLASS(SpringForceField)

using namespace sofa::defaulttype;


//Register in the Factory
int SpringForceFieldClass = core::RegisterObject("Springs")
#ifndef SOFA_FLOAT
        .add< SpringForceField<Vec3dTypes> >()
        .add< SpringForceField<Vec2dTypes> >()
        .add< SpringForceField<Vec1dTypes> >()
        .add< SpringForceField<Vec6dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< SpringForceField<Vec3fTypes> >()
        .add< SpringForceField<Vec2fTypes> >()
        .add< SpringForceField<Vec1fTypes> >()
        .add< SpringForceField<Vec6fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_DEFORMABLE_API LinearSpring<double>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec3dTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec2dTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec1dTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec6dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_DEFORMABLE_API LinearSpring<float>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec3fTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec2fTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec1fTypes>;
template class SOFA_DEFORMABLE_API SpringForceField<Vec6fTypes>;
#endif


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

