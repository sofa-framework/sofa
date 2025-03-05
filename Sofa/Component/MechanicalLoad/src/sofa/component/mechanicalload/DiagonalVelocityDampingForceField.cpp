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
#define SOFA_COMPONENT_FORCEFIELD_DIAGONALVELOCITYDAMPINGFORCEFIELD_CPP

#include <sofa/component/mechanicalload/DiagonalVelocityDampingForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::mechanicalload
{

using namespace sofa::defaulttype;

void registerDiagonalVelocityDampingForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Diagonal velocity damping.")
        .add< DiagonalVelocityDampingForceField<Vec3Types> >()
        .add< DiagonalVelocityDampingForceField<Vec2Types> >()
        .add< DiagonalVelocityDampingForceField<Vec1Types> >()
        .add< DiagonalVelocityDampingForceField<Vec6Types> >()
        .add< DiagonalVelocityDampingForceField<Rigid3Types> >()
        .add< DiagonalVelocityDampingForceField<Rigid2Types> >());
}

template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Vec3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Vec2Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Vec1Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Vec6Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Rigid3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API DiagonalVelocityDampingForceField<Rigid2Types>;

} // namespace sofa::component::mechanicalload
