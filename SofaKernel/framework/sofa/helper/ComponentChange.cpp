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

};

std::map<std::string, ComponentChange> uncreatableComponents = {
    // SofaDistanceGrid was pluginized in #389
    {"BarycentricPenalityContact", Pluginized("v17.12", "SofaDistanceGrid")},
    {"DistanceGridCollisionModel", Pluginized("v17.12", "SofaDistanceGrid")},
    {"FFDDistanceGridDiscreteIntersection", Pluginized("v17.12", "SofaDistanceGrid")},
    {"RayDistanceGridContact", Pluginized("v17.12", "SofaDistanceGrid")},
    {"RigidDistanceGridDiscreteIntersection", Pluginized("v17.12", "SofaDistanceGrid")},
    {"DistanceGridForceField", Pluginized("v17.12", "SofaDistanceGrid")},

    // SofaImplicitField was pluginized in #389
    {"ImplicitSurfaceContainer", Pluginized("v17.12", "SofaImplicitField")},
    {"InterpolatedImplicitSurface", Pluginized("v17.12", "SofaImplicitField")},
    {"SphereSurface", Pluginized("v17.12", "SofaImplicitField")},
    {"ImplicitSurfaceMapping", Pluginized("v17.12", "SofaImplicitField")},

    // SofaPreconditioner was pluginized in #663
    {"ShewchukPCGLinearSolver", Pluginized("v18.06", "SofaPreconditioner")},
    {"JacobiPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"BlockJacobiPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"SSORPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"WarpPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},
    {"PrecomputedWarpPreconditioner", Pluginized("v18.06", "SofaPreconditioner")},

    // SofaSparseSolver was pluginized in #663
    {"PrecomputedLinearSolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseCholeskySolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseLUSolver", Pluginized("v18.06", "SofaSparseSolver")},
    {"SparseLDLSolver", Pluginized("v18.06", "SofaSparseSolver")},

    // SofaExporter was pluginized in #915
    {"WriteTopology", Pluginized("v19.06", "SofaExporter")},
    {"MeshExporter", Pluginized("v19.06", "SofaExporter")},
    {"OBJExporter", Pluginized("v19.06", "SofaExporter")},
    {"STLExporter", Pluginized("v19.06", "SofaExporter")},
    {"VTKExporter", Pluginized("v19.06", "SofaExporter")},
    {"WriteState", Pluginized("v19.06", "SofaExporter")},
    {"WriteTopology", Pluginized("v19.06", "SofaExporter")},

    // SofaHaptics was pluginized in #945
    {"NullForceFeedback", Pluginized("v19.06", "SofaHaptics")},
    {"LCPForceFeedback", Pluginized("v19.06", "SofaHaptics")},

    // SofaOpenglVisual was pluginized in #1080
    {"ClipPlane", Pluginized("v19.06", "SofaOpenglVisual")},
    {"CompositingVisualLoop", Pluginized("v19.06", "SofaOpenglVisual")},
    {"DataDisplay", Pluginized("v19.06", "SofaOpenglVisual")},
    {"Light", Pluginized("v19.06", "SofaOpenglVisual")},
    {"LightManager", Pluginized("v19.06", "SofaOpenglVisual")},
    {"MergeVisualModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglAttribute", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglColorMap", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglCylinderModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglGrid", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglLabel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglLineAxis", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglOITShader", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglRenderingSRGB", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglSceneFrame", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglShader", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglShaderMacro", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglShaderVisualModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglShadowShader", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglTexture", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglTexturePointer", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglVariable", Pluginized("v19.06", "SofaOpenglVisual")},
    {"OglViewport", Pluginized("v19.06", "SofaOpenglVisual")},
    {"PointSplatModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"PostProcessManager", Pluginized("v19.06", "SofaOpenglVisual")},
    {"SlicedVolumetricModel", Pluginized("v19.06", "SofaOpenglVisual")},
    {"VisualManagerPass", Pluginized("v19.06", "SofaOpenglVisual")},
    {"VisualmanagerSecondaryPass", Pluginized("v19.06", "SofaOpenglVisual")},

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

