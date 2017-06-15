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
#include <sofa/helper/system/config.h>
#include <SofaUserInteraction/initUserInteraction.h>


namespace sofa
{

namespace component
{


void initUserInteraction()
{
    static bool first = true;
    if (first)
    {
        first = false;
    }
}

SOFA_LINK_CLASS(RayTraceDetection)
SOFA_LINK_CLASS(RayContact)
SOFA_LINK_CLASS(MouseInteractor)
SOFA_LINK_CLASS(ArticulatedHierarchyController)
SOFA_LINK_CLASS(ArticulatedHierarchyBVHController)
SOFA_LINK_CLASS(EdgeSetController)
SOFA_LINK_CLASS(MechanicalStateController)
SOFA_LINK_CLASS(MechanicalStateControllerOmni)
SOFA_LINK_CLASS(Ray)
SOFA_LINK_CLASS(RayDiscreteIntersection)
SOFA_LINK_CLASS(RayNewProximityIntersection)
SOFA_LINK_CLASS(NodeToggleController)
SOFA_LINK_CLASS(GraspingManager)
SOFA_LINK_CLASS(InterpolationController)


} // namespace component

} // namespace sofa
