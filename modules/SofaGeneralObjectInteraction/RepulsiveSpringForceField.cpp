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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_REPULSIVESPRINGFORCEFIELD_CPP
#include <SofaGeneralObjectInteraction/RepulsiveSpringForceField.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;

SOFA_DECL_CLASS(RepulsiveSpringForceField)

// Register in the Factory
int RepulsiveSpringForceFieldClass = core::RegisterObject("Springs which only repell")
#ifndef SOFA_FLOAT
        .add< RepulsiveSpringForceField<Vec3dTypes> >()
        .add< RepulsiveSpringForceField<Vec2dTypes> >()
        .add< RepulsiveSpringForceField<Vec1dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< RepulsiveSpringForceField<Vec3fTypes> >()
        .add< RepulsiveSpringForceField<Vec2fTypes> >()
        .add< RepulsiveSpringForceField<Vec1fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class RepulsiveSpringForceField<Vec3dTypes>;
template class RepulsiveSpringForceField<Vec2dTypes>;
template class RepulsiveSpringForceField<Vec1dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class RepulsiveSpringForceField<Vec3fTypes>;
template class RepulsiveSpringForceField<Vec2fTypes>;
template class RepulsiveSpringForceField<Vec1fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa

