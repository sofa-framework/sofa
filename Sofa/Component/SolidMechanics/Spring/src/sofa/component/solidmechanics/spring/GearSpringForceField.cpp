/******************************************************************************
*                 SOFA, Simulation Open-Framework Architecture                *
*                    (c) 2006 INRIA, USTL, UJF, CNRS, MGH                     *
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
#define SOFA_COMPONENT_FORCEFIELD_GEARSPRINGFORCEFIELD_CPP

#include <sofa/component/solidmechanics/spring/GearSpringForceField.inl>
#include <sofa/core/behavior/PairInteractionForceField.inl>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/core/behavior/MechanicalState.h>
#include <sofa/core/ObjectFactory.h>


namespace sofa::component::solidmechanics::spring
{

using namespace sofa::defaulttype;

void registerGearSpringForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Gear springs for Rigids.")
        .add< GearSpringForceField<Rigid3Types> >());
}

template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API GearSpring<defaulttype::Rigid3Types>;
template class SOFA_COMPONENT_SOLIDMECHANICS_SPRING_API GearSpringForceField<defaulttype::Rigid3Types>;

} // namespace sofa::component::solidmechanics::spring

