/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2018 INRIA, USTL, UJF, CNRS, MGH                    *
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
#include <SofaHaptics/NullForceFeedbackT.h>
#include <sofa/core/ObjectFactory.h>

using namespace std;

namespace sofa
{
namespace component
{
namespace controller
{

//void NullForceFeedback::init()
//{
//	this->ForceFeedback::init();
//};
//
//void NullForceFeedback::computeForce(SReal /*x*/, SReal /*y*/, SReal /*z*/, SReal /*u*/, SReal /*v*/, SReal /*w*/, SReal /*q*/, SReal& fx, SReal& fy, SReal& fz)
//{
//	fx = fy = fz = 0.0;
//};
//
//void NullForceFeedback::computeWrench(const SolidTypes<SReal>::Transform &/*world_H_tool*/, const SolidTypes<SReal>::SpatialVector &/*V_tool_world*/, SolidTypes<SReal>::SpatialVector &W_tool_world )
//{
//	W_tool_world.clear();
//};
int nullForceFeedbackTClass = sofa::core::RegisterObject("Null force feedback for haptic feedback device")
#ifndef SOFA_FLOAT
        .add< NullForceFeedbackT<sofa::defaulttype::Vec1dTypes> >()
        .add< NullForceFeedbackT<sofa::defaulttype::Rigid3dTypes> >()
#endif
#ifndef SOFA_DOUBLE
        .add< NullForceFeedbackT<sofa::defaulttype::Vec1fTypes> >()
        .add< NullForceFeedbackT<sofa::defaulttype::Rigid3fTypes> >()
#endif
        ;

//int nullForceFeedbackClass = sofa::core::RegisterObject("Null force feedback for haptic feedback device")
//    .add< NullForceFeedback >();

SOFA_DECL_CLASS(NullForceFeedbackT)

} // namespace controller
} // namespace component
} // namespace sofa
