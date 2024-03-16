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
#pragma once

#include <sofa/component/haptics/config.h>

#include <sofa/component/haptics/ForceFeedback.h>

namespace sofa::component::haptics
{


/// @brief Null force feedback for haptic feedback device
class SOFA_COMPONENT_HAPTICS_API NullForceFeedback : public ForceFeedback
{
public:
    SOFA_CLASS(NullForceFeedback,ForceFeedback);
    void init() override;

    void computeForce(SReal x, SReal y, SReal z, SReal u, SReal v, SReal w, SReal q, SReal& fx, SReal& fy, SReal& fz) override;
    void computeWrench(const sofa::type::Transform<SReal> &world_H_tool, const sofa::type::SpatialVector<SReal> &V_tool_world, sofa::type::SpatialVector<SReal> &W_tool_world ) override;
};

} // namespace sofa::component::haptics
