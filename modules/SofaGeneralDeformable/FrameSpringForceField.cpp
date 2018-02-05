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
#define SOFA_COMPONENT_INTERACTIONFORCEFIELD_FRAMESPRINGFORCEFIELD_CPP
#include <SofaGeneralDeformable/FrameSpringForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/RigidTypes.h>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS ( FrameSpringForceField )
// Register in the Factory

int FrameSpringForceFieldClass = core::RegisterObject ( "Springs for Flexibles" )
#ifndef SOFA_FLOAT
        .add< FrameSpringForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< FrameSpringForceField<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_GENERAL_DEFORMABLE_API FrameSpringForceField<Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_GENERAL_DEFORMABLE_API FrameSpringForceField<Rigid3fTypes>;
#endif

} // namespace interactionforcefield

} // namespace component

} // namespace sofa
