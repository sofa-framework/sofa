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
#define SOFA_COMPONENT_FORCEFIELD_NODALLINEARDAMPINGFORCEFIELD_CPP

#include <sofa/component/mechanicalload/NodalLinearDampingForceField.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::mechanicalload
{

using namespace sofa::defaulttype;

int NodalLinearDampingForceFieldClass = core::RegisterObject("Linear damping force applied on the degrees of freedom")
.add< NodalLinearDampingForceField<Vec3Types> >()
.add< NodalLinearDampingForceField<Vec2Types> >()
.add< NodalLinearDampingForceField<Vec1Types> >()
.add< NodalLinearDampingForceField<Vec6Types> >()
.add< NodalLinearDampingForceField<Rigid3Types> >()
.add< NodalLinearDampingForceField<Rigid2Types> >()

;

template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Vec3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Vec2Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Vec1Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Vec6Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Rigid3Types>;
template class SOFA_COMPONENT_MECHANICALLOAD_API NodalLinearDampingForceField<Rigid2Types>;

} // namespace sofa::component::mechanicalload
