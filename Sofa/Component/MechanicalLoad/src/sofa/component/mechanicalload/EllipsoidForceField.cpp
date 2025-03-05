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
#define SOFA_COMPONENT_FORCEFIELD_ELLIPSOIDFORCEFIELD_CPP

#include <sofa/component/mechanicalload/EllipsoidForceField.inl>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa::component::mechanicalload
{

using namespace sofa::defaulttype;

void registerEllipsoidForceField(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(core::ObjectRegistrationData("Outward / inward repulsion applied by an ellipsoid geometry.")
        .add< EllipsoidForceField<Vec3Types> >()
        .add< EllipsoidForceField<Vec2Types> >()
        .add< EllipsoidForceField<Vec1Types> >());
}

template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<Vec3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<Vec2Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API EllipsoidForceField<Vec1Types>;

} // namespace sofa::component::mechanicalload
