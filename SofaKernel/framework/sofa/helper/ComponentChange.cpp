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
#include "ComponentChange.h"


namespace sofa
{
namespace helper
{
namespace lifecycle
{

std::map<std::string, Deprecated> deprecatedComponents = {
    // SofaBoundaryCondition
    {"BuoyantForceField", Deprecated("v17.12", "v18.12")},
    {"VaccumSphereForceField", Deprecated("v17.12", "v18.12")},

    // SofaMiscForceField
    {"ForceMaskOff", Deprecated("v17.12", "v18.12")},
    {"LineBendingSprings", Deprecated("v17.12", "v18.12")},
    {"WashingMachineForceField", Deprecated("v17.12", "v18.12")},
    {"LennardJonesForceField", Deprecated("v17.12", "v18.12")},

    // SofaMiscMapping
    {"CatmullRomSplineMapping", Deprecated("v17.12", "v18.12")},
    {"CenterPointMechanicalMapping", Deprecated("v17.12", "v18.12")},
    {"CurveMapping", Deprecated("v17.12", "v18.12")},
    {"ExternalInterpolationMapping", Deprecated("v17.12", "v18.12")},
    {"ProjectionToLineMapping", Deprecated("v17.12", "v18.12")},
    {"ProjectionToPlaneMapping", Deprecated("v17.12", "v18.12")},

    // SofaMisc
    {"ParallelCGLinearSolver", Deprecated("v17.12", "v18.12")},

    // SofaOpenglVisual
    {"OglCylinderModel", Deprecated("v17.12", "v18.12")},
    {"OglGrid", Deprecated("v17.12", "v18.12")},
    {"OglRenderingSRGB", Deprecated("v17.12", "v18.12")},
    {"OglLineAxis", Deprecated("v17.12", "v18.12")},
    {"OglSceneFrame", Deprecated("v17.12", "v18.12")},

    // SofaUserInteraction
    {"ArticulatedHierarchyBVHController", Deprecated("v17.12", "v18.12")},
    {"ArticulatedHierarchyController", Deprecated("v17.12", "v18.12")},
    {"DisabledContact", Deprecated("v17.12", "v18.12")},
    {"EdgeSetController", Deprecated("v17.12", "v18.12")},
    {"GraspingManager", Deprecated("v17.12", "v18.12")},
    {"InterpolationController", Deprecated("v17.12", "v18.12")},
    {"MechanicalStateControllerOmni", Deprecated("v17.12", "v18.12")},
    {"NodeToggleController", Deprecated("v17.12", "v18.12")},
};

std::map<std::string, ComponentChange> uncreatableComponents = {
    {"BarycentricPenalityContact", Pluginized("v17.12", "SofaDistanceGrid")},
    {"DistanceGridCollisionModel", Pluginized("v17.12", "SofaDistanceGrid")},
    {"FFDDistanceGridDiscreteIntersection", Pluginized("v17.12", "SofaDistanceGrid")},
    {"RayDistanceGridContact", Pluginized("v17.12", "SofaDistanceGrid")},
    {"RigidDistanceGridDiscreteIntersection", Pluginized("v17.12", "SofaDistanceGrid")},
    {"DistanceGridForceField", Pluginized("v17.12", "SofaDistanceGrid")},

    {"ImplicitSurfaceContainer", Pluginized("v17.12", "SofaImplicitField")},
    {"InterpolatedImplicitSurface", Pluginized("v17.12", "SofaImplicitField")},
    {"SphereSurface", Pluginized("v17.12", "SofaImplicitField")},
    {"ImplicitSurfaceMapping", Pluginized("v17.12", "SofaImplicitField")},
};


} // namespace lifecycle
} // namespace helper
} // namespace sofa

