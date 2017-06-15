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
#define SOFA_COMPONENT_FORCEFIELD_GEARSPRINGFORCEFIELD_CPP
#include <SofaMiscForceField/GearSpringForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>
//#include <typeinfo>


namespace sofa
{

namespace component
{

namespace interactionforcefield
{

using namespace sofa::defaulttype;


SOFA_DECL_CLASS(GearSpringForceField)

// Register in the Factory
int GearSpringForceFieldClass = core::RegisterObject("Gear springs for Rigids")
#ifndef SOFA_FLOAT
        .add< GearSpringForceField<Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< GearSpringForceField<Rigid3fTypes> >()
#endif
        ;

#ifndef SOFA_FLOAT
template class SOFA_MISC_FORCEFIELD_API GearSpring<defaulttype::Rigid3dTypes>;
template class SOFA_MISC_FORCEFIELD_API GearSpringForceField<defaulttype::Rigid3dTypes>;
#endif
#ifndef SOFA_DOUBLE
template class SOFA_MISC_FORCEFIELD_API GearSpring<defaulttype::Rigid3fTypes>;
template class SOFA_MISC_FORCEFIELD_API GearSpringForceField<defaulttype::Rigid3fTypes>;
#endif


} // namespace interactionforcefield

} // namespace component

} // namespace sofa

