/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, development version     *
*                (c) 2006-2019 INRIA, USTL, UJF, CNRS, MGH                    *
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
    // SofaMiscForceField
    {"LennardJonesForceField", Deprecated("v17.12", "v18.12")},

    // SofaOpenglVisual
    {"OglCylinderModel", Deprecated("v17.12", "v18.12")},
    {"OglGrid", Deprecated("v17.12", "v18.12")},
    {"OglRenderingSRGB", Deprecated("v17.12", "v18.12")},
    {"OglLineAxis", Deprecated("v17.12", "v18.12")},
    {"OglSceneFrame", Deprecated("v17.12", "v18.12")},
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

    // Pluginized in #663
    {"ShewchukPCGLinearSolver", Pluginized("v18.06", "SofaPreconditioner")},
    {"JacobiPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"BlockJacobiPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"SSORPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"WarpPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"PrecomputedWarpPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},

    // Pluginized in #663
    {"PrecomputedLinearSolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseCholeskySolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseLUSolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseLDLSolver", Pluginized("v18.06", "SofaSparseSolver")},

    /***********************/
    // REMOVED SINCE v18.12

    // SofaBoundaryCondition
    {"BuoyantForceField", Removed("v17.12", "v18.12")},
    {"VaccumSphereForceField", Removed("v17.12", "v18.12")},

    // SofaMiscForceField
    {"ForceMaskOff", Removed("v17.12", "v18.12")},
    {"LineBendingSprings", Removed("v17.12", "v18.12")},
    {"WashingMachineForceField", Removed("v17.12", "v18.12")},

    // SofaMiscMapping
    {"CatmullRomSplineMapping", Removed("v17.12", "v18.12")},
    {"CenterPointMechanicalMapping", Removed("v17.12", "v18.12")},
    {"CurveMapping", Removed("v17.12", "v18.12")},
    {"ExternalInterpolationMapping", Removed("v17.12", "v18.12")},
    {"ProjectionToLineMapping", Removed("v17.12", "v18.12")},
    {"ProjectionToPlaneMapping", Removed("v17.12", "v18.12")},

    // SofaMisc
    {"ParallelCGLinearSolver", Removed("v17.12", "v18.12")},

    // SofaUserInteraction
    {"ArticulatedHierarchyBVHController", Removed("v17.12", "v18.12")},
    {"ArticulatedHierarchyController", Removed("v17.12", "v18.12")},
    {"DisabledContact", Removed("v17.12", "v18.12")},
    {"EdgeSetController", Removed("v17.12", "v18.12")},
    {"GraspingManager", Removed("v17.12", "v18.12")},
    {"InterpolationController", Removed("v17.12", "v18.12")},
    {"MechanicalStateControllerOmni", Removed("v17.12", "v18.12")},
    {"NodeToggleController", Removed("v17.12", "v18.12")},

};


} // namespace lifecycle
} // namespace helper
} // namespace sofa

