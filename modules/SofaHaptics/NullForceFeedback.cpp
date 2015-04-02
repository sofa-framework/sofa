/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <SofaHaptics/NullForceFeedback.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/core/ObjectFactory.h>

using namespace std;

namespace sofa
{
namespace component
{
namespace controller
{

void NullForceFeedback::init()
{
    this->ForceFeedback::init();
};

void NullForceFeedback::computeForce(SReal /*x*/, SReal /*y*/, SReal /*z*/, SReal /*u*/, SReal /*v*/, SReal /*w*/, SReal /*q*/, SReal& fx, SReal& fy, SReal& fz)
{
    fx = fy = fz = 0.0;
};

void NullForceFeedback::computeWrench(const sofa::defaulttype::SolidTypes<SReal>::Transform &/*world_H_tool*/, const sofa::defaulttype::SolidTypes<SReal>::SpatialVector &/*V_tool_world*/, sofa::defaulttype::SolidTypes<SReal>::SpatialVector &W_tool_world )
{
    W_tool_world.clear();
};

int nullForceFeedbackClass = sofa::core::RegisterObject("Null force feedback for haptic feedback device")
        .add< NullForceFeedback >();

SOFA_DECL_CLASS(NullForceFeedback)

} // namespace controller
} // namespace component
} // namespace sofa
