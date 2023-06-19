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
#define SOFA_COMPONENT_CONTROLLER_LCPFORCEFEEDBACK_CPP

#include <sofa/component/haptics/LCPForceFeedback.inl>
#include <sofa/core/ObjectFactory.h>
#include <sofa/defaulttype/RigidTypes.h>

namespace sofa::component::haptics
{

using sofa::defaulttype::Rigid3Types;

template <>
void LCPForceFeedback< Rigid3Types >::computeForce(SReal x, SReal y, SReal z, SReal, SReal, SReal, SReal, SReal& fx, SReal& fy, SReal& fz)
{
    Rigid3Types::VecCoord state;
    Rigid3Types::VecDeriv forces;
    state.resize(1);
    state[0].getCenter() = sofa::type::Vec3(x,y,z);
    computeForce(state,forces);
    fx = getVCenter(forces[0]).x();
    fy = getVCenter(forces[0]).y();
    fz = getVCenter(forces[0]).z();
}


template <>
void LCPForceFeedback< Rigid3Types >::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &world_H_tool,
        const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &/*V_tool_world*/,
        sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world )
{
    if (!this->d_activate.getValue())
    {
        return;
    }

    Rigid3Types::VecCoord state;
    Rigid3Types::VecDeriv forces;
    state.resize(1);
    state[0].getCenter()	  = world_H_tool.getOrigin();
    state[0].getOrientation() = world_H_tool.getOrientation();

    computeForce(state,forces);

    W_tool_world.setForce(getVCenter(forces[0]));
    W_tool_world.setTorque(getVOrientation(forces[0]));
}

int lCPForceFeedbackClass = sofa::core::RegisterObject("LCP force feedback for the device")
        .add< LCPForceFeedback<defaulttype::Vec1Types> >()
        .add< LCPForceFeedback<defaulttype::Rigid3Types> >();

template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Vec1Types>;
template class SOFA_COMPONENT_HAPTICS_API LCPForceFeedback<defaulttype::Rigid3Types>;


} // namespace sofa::component::haptics
