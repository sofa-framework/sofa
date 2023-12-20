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
#include <sofa/helper/ComponentChange.h>


namespace sofa::helper::lifecycle
{

const std::map<std::string, Deprecated, std::less<> > deprecatedComponents = {
    {"RayTraceDetection", Deprecated("v21.06", "v21.12")},
    {"BruteForceDetection", Deprecated("v21.06", "v21.12")},
    {"DirectSAP", Deprecated("v21.06", "v21.12")},
    {"RigidRigidMapping", Deprecated("v23.06", "v23.12", "You can use the component RigidMapping with template='Rigid3,Rigid3' instead.")},
};

const std::map<std::string, ComponentChange, std::less<> > movedComponents = {
    // SofaValidation was pluginized in #1302
    {"CompareState", Pluginized("v20.06", "SofaValidation")},
    {"CompareTopology", Pluginized("v20.06", "SofaValidation")},
    {"DataController", Pluginized("v20.06", "SofaValidation")},
    {"DataMonitor", Pluginized("v20.06", "SofaValidation")},
    {"DevAngleCollisionMonitor", Pluginized("v20.06", "SofaValidation")},
    {"DevMonitorManager", Pluginized("v20.06", "SofaValidation")},
    {"DevTensionMonitor", Pluginized("v20.06", "SofaValidation")},
    {"EvalPointsDistance", Pluginized("v20.06", "SofaValidation")},
    {"EvalSurfaceDistance", Pluginized("v20.06", "SofaValidation")},
    {"ExtraMonitor", Pluginized("v20.06", "SofaValidation")},
    {"Monitor", Pluginized("v20.06", "SofaValidation")},

    // SofaGraphComponent was pluginized in #1531
    { "Gravity", Pluginized("v20.12", "SofaGraphComponent") },
    { "PauseAnimationOnEvent", Pluginized("v20.12", "SofaGraphComponent") },

    // SofaUserInteraction was pluginized in #1588
    { "SleepController", Pluginized("v20.12", "SofaUserInteraction") },

    // LMConstraint was pluginized in #1659
    { "BaseLMConstraint", Pluginized("v20.12", "LMConstraint") },
    { "LMConstraint", Pluginized("v20.12", "LMConstraint") },
    { "TetrahedronBarycentricDistanceLMConstraintContact", Pluginized("v20.12", "LMConstraint") },
    { "BarycentricDistanceLMConstraintContact_DistanceGrid", Pluginized("v20.12", "LMConstraint") },
    { "BarycentricDistanceLMConstraintContact", Pluginized("v20.12", "LMConstraint") },
    { "DistanceLMConstraint", Pluginized("v20.12", "LMConstraint") },
    { "DistanceLMContactConstraint", Pluginized("v20.12", "LMConstraint") },
    { "DOFBlockerLMConstraint", Pluginized("v20.12", "LMConstraint") },
    { "FixedLMConstraint", Pluginized("v20.12", "LMConstraint") },
    { "LMConstraintSolver", Pluginized("v20.12", "LMConstraint") },
    { "LMConstraintDirectSolver", Pluginized("v20.12", "LMConstraint") },

    /***********************/
    // MOVED SINCE v21.06
    { "SpatialGridPointModel", Moved("v21.06", "SofaMiscCollision", "SofaSphFluid") },

    // MOVED SINCE v21.12
    { "LULinearSolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },
    //{"NewMatCholeskySolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },
    //{"NewMatCGLinearSolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },

    // MOVED SINCE v22.06
    { "GlobalSystemMatrixExporter", Moved("v22.06", "SofaBaseLinearSolver", "SofaMatrix") },

    // SofaMiscSolver was deprecated in #2571
    { "DampVelocitySolver", Moved("v22.06", "SofaMiscSolver", "Sofa.Component.ODESolver.Forward") },
    { "NewmarkImplicitSolver", Moved("v22.06", "SofaMiscSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaExplicitOdeSolver was deprecated in #2571
    { "EulerExplicitSolver", Moved("v22.06", "SofaExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },

    // SofaImplicitOdeSolver was deprecated in #2571
    { "EulerImplicitSolver", Moved("v22.06", "SofaImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },
    { "StaticSolver", Moved("v22.06", "SofaImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaGeneralExplicitOdeSolver was deprecated in #2571
    { "CentralDifferenceSolver", Moved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },
    { "RungeKutta2Solver", Moved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },
    { "RungeKutta4Solver", Moved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },

    // SofaGeneralImplicitOdeSolver was deprecated in #2571
    { "VariationalSymplecticSolver", Moved("v22.06", "SofaGeneralImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaLoader was deprecated in #2582
    { "MeshOBJLoader", Moved("v22.06", "SofaLoader", "Sofa.Component.IO.Mesh") },
    { "MeshVTKLoader", Moved("v22.06", "SofaLoader", "Sofa.Component.IO.Mesh") },

    // SofaGeneralLoader was deprecated in #2582
    { "MeshGmshLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "GIDMeshLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "GridMeshCreator", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshOffLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshSTLLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshTrianLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshXspLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "OffSequenceLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "SphereLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "StringMeshCreator", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "VoxelGridLoader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "ReadState", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },
    { "ReadTopology", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },
    { "InputEventReader", Moved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },

    // SofaExporter was deprecated in #2582
    { "BlenderExporter", Moved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "MeshExporter", Moved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "STLExporter", Moved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "VisualModelOBJExporter", Moved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "VTKExporter", Moved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "WriteState", Moved("v22.06", "SofaExporter", "Sofa.Component.Playback") },
    { "WriteTopology", Moved("v22.06", "SofaExporter", "Sofa.Component.Playback") },

    // SofaBaseUtils was deprecated in #2605
    { "AddResourceRepository", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MakeAliasComponent", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MakeDataAliasComponent", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MessageHandlerComponent", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "FileMessageHandlerComponent", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "InfoComponent", Moved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "RequiredPlugin", Moved("v22.06", "SofaBaseUtils", "Sofa.Core") },

    // SofaGraphComponent was deprecated in #2605, #2843 and #2895
    { "AddFrameButtonSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.GUI.Component") },
    { "AddRecordedCameraButtonSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.GUI.Component") },
    { "StartNavigationButtonSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.Component.Setting") },
    { "AttachBodyButtonSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.GUI.Component") },
    { "FixPickedParticleButtonSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.GUI.Component") },
    { "SofaDefaultPathSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.Component.Setting") },
    { "StatsSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.Component.Setting") },
    { "ViewerSetting", Moved("v22.06", "SofaGraphComponent", "Sofa.Component.Setting") },
    { "APIVersion", Moved("v22.06", "SofaGraphComponent", "Sofa.Component.Setting") },

    // SofaBaseTopology was deprecated in #2612
    { "EdgeSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetGeometryAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyAlgorithms", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyContainer", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyModifier", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "MeshTopology", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Constant") },
    { "GridTopology", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },
    { "RegularGridTopology", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },
    { "SparseGridTopology", Moved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },

    // SofaGeneralTopology was deprecated in #2612
    { "CubeTopology", Moved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Constant") },
    { "SphereQuadTopology", Moved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Constant") },
    { "CylinderGridTopology", Moved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Grid") },
    { "SphereGridTopology", Moved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Grid") },

    // SofaNonUniformFem was deprecated in #2612 and #2759
    { "DynamicSparseGridGeometryAlgorithms", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyAlgorithms", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyContainer", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyModifier", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "MultilevelHexahedronSetTopologyContainer", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "SparseGridMultipleTopology", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Grid") },
    { "SparseGridRamificationTopology", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Grid") },
    { "NonUniformHexahedralFEMForceFieldAndMass", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "NonUniformHexahedronFEMForceFieldAndMass", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "HexahedronCompositeFEMForceFieldAndMass", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "HexahedronCompositeFEMMapping", Moved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },

    // SofaTopologicalMapping was deprecated in #2612 and #XXXX
    { "CenterPointTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Edge2QuadTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Hexa2QuadTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Hexa2TetraTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "IdentityTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Mesh2PointTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Quad2TriangleTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SimpleTesselatedHexaTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SimpleTesselatedTetraTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SubsetTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Tetra2TriangleTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Triangle2EdgeTopologicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Mesh2PointMechanicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Mapping.Linear") },
    { "SimpleTesselatedTetraMechanicalMapping", Moved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Mapping.Linear") },

    // SofaMiscTopology was deprecated in #2612
    { "TopologicalChangeProcessor", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },
    { "TopologyBoundingTrasher", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },
    { "TopologyChecker", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },

    // SofaBaseVisual was deprecated in #2679 and #XXXX
    { "Camera", Moved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "InteractiveCamera", Moved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "VisualModelImpl", Moved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "VisualStyle", Moved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "BackgroundSetting", Moved("v22.06", "SofaBaseVisual", "Sofa.Component.UserInterface.ConfigurationSetting") },

    // SofaGeneralVisual was deprecated in #2679
    { "RecordedCamera", Moved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },
    { "Visual3DText", Moved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },
    { "VisualTransform", Moved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },

    // SofaSimpleFem was deprecated in #2753 and ....
    { "TetrahedronDiffusionFEMForceField", Moved("v22.06", "SofaSimpleFem", "Sofa.Component.Diffusion") },

    // SofaOpenglVisual was deprecated in #2709
    { "OglSceneFrame", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "DataDisplay", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "MergeVisualModels", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglCylinderModel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglModel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "PointSplatModel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "SlicedVolumetricModel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglColorMap", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "OglLabel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "OglViewport", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "ClipPlane", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "CompositingVisualLoop", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "DirectionalLight", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "PositionalLight", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "SpotLight", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "LightManager", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatAttribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntAttribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUIntAttribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt2Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt3Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt4Attribute", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglOITShader", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglRenderingSRGB", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShader", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShaderMacro", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShaderVisualModel", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShadowShader", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglTexture", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglTexture2D", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntVectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatVectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2x3Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3x2Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2x4Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4x2Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3x4Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4x3Variable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4VectorVariable", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OrderIndependentTransparencyManager", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "PostProcessManager", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "VisualManagerPass", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "VisualManagerSecondaryPass", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "TextureInterpolation", Moved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Engine") },

    // SofaBaseLinearSolver was deprecated in #2717
    { "CGLinearSolver", Moved("v22.06", "SofaBaseLinearSolver", "Sofa.Component.LinearSolver.Iterative") },

    // SofaGeneralLinearSolver was deprecated in #2717
    { "MinResLinearSolver", Moved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Iterative") },
    { "BTDLinearSolver", Moved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "CholeskySolver", Moved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },

    // SofaSparseSolver was deprecated in #2717
    { "FillReducingOrdering", Moved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "PrecomputedLinearSolver", Moved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "SparseLDLSolver", Moved("v22.06", "SofaSparseSolver", "Sofa.Component.LinearSolver.Direct") },

    // SofaDenseSolver was deprecated in #2717
    { "SVDLinearSolver", Moved("v22.06", "SofaDenseSolver", "Sofa.Component.LinearSolver.Direct") },

    // SofaPreconditioner was deprecated in #2717
    { "ShewchukPCGLinearSolver", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Iterative") },
    { "JacobiPreconditioner", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "BlockJacobiPreconditioner", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "PrecomputedWarpPreconditioner", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "SSORPreconditioner", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "WarpPreconditioner", Moved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },

    // SofaBaseMechanics was deprecated in #2752, #2635 and #2766
    { "DiagonalMass", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mass") },
    { "UniformMass", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mass") },
    { "BarycentricMapping", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "IdentityMapping", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "SubsetMapping", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "MechanicalObject", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.StateContainer") },
    { "MappedObject", Moved("v22.06", "SofaBaseMechanics", "Sofa.Component.StateContainer") },

    // SofaMiscForceField was deprecated in #2752 and ...
    { "MeshMatrixMass", Moved("v22.06", "SofaMiscForceField", "Sofa.Component.Mass") },
    { "GearSpringForceField", Moved("v22.06", "SofaMiscForceField", "Sofa.Component.SolidMechanics.Spring") },


    // SofaRigid was deprecated in #2635 and #2759
    { "RigidMapping", Moved("v22.06", "SofaRigid", "Sofa.Component.Mapping.NonLinear") },
    { "RigidRigidMapping", Moved("v22.06", "SofaRigid", "Sofa.Component.Mapping.NonLinear") },
    { "JointSpringForceField", Moved("v22.06", "SofaRigid", "Sofa.Component.SolidMechanics.Spring") },

    // Movedgid was deprecated in #2635 and ...
    { "LineSetSkinningMapping", Moved("v22.06", "Movedgid", "Sofa.Component.Mapping.Linear") },
    { "SkinningMapping", Moved("v22.06", "Movedgid", "Sofa.Component.Mapping.Linear") },
    { "ArticulatedHierarchyContainer", Moved("v22.06", "Movedgid", "ArticulatedSystemPlugin") },
    { "ArticulationCenter", Moved("v22.06", "Movedgid", "ArticulatedSystemPlugin") },
    { "Articulation", Moved("v22.06", "Movedgid", "ArticulatedSystemPlugin") },
    { "ArticulatedSystemMapping", Moved("v22.06", "Movedgid", "ArticulatedSystemPlugin") },

    // SofaMiscMapping was deprecated in #2635
    { "BeamLinearMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMulti2Mapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMultiMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "DeformableOnRigidFrameMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "DistanceFromTargetMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "DistanceMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "IdentityMultiMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "SquareMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "SquareDistanceMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "SubsetMultiMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "TubularMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "VoidMapping", Moved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },

    // SofaConstraint was deprecated in #2635, #2790, #2796, #2813 and ...
    { "BilateralInteractionConstraint", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "GenericConstraintCorrection", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "GenericConstraintSolver", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Solver") },
    { "LCPConstraintSolver", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Solver") },
    { "LinearSolverConstraintCorrection", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "PrecomputedConstraintCorrection", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "SlidingConstraint", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "StopperConstraint", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "UncoupledConstraintCorrection", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "UniformConstraint", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "UnilateralInteractionConstraint", Moved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "ConstraintAnimationLoop", Moved("v22.06", "SofaConstraint", "Sofa.Component.AnimationLoop") },
    { "FreeMotionAnimationLoop", Moved("v22.06", "SofaConstraint", "Sofa.Component.AnimationLoop") },
    { "LocalMinDistance", Moved("v22.06", "SofaConstraint", "Sofa.Component.Collision.Detection.Intersection") },

    // SofaGeneralAnimationLoop was deprecated in #2635 and #2796
    { "MultiStepAnimationLoop", Moved("v22.06", "SofaGeneralAnimationLoop", "Sofa.Component.AnimationLoop") },
    { "MultiTagAnimationLoop", Moved("v22.06", "SofaGeneralAnimationLoop", "Sofa.Component.AnimationLoop") },

    // SofaSimpleFem was deprecated in #2759
    { "HexahedronFEMForceField", Moved("v22.06", "SofaSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TetrahedronFEMForceField", Moved("v22.06", "SofaSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },

    // SofaGeneralSimpleFem was deprecated in #2759
    { "BeamFEMForceField", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedralFEMForceField", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedralFEMForceFieldAndMass", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedronFEMForceFieldAndMass", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TetrahedralCorotationalFEMForceField", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularFEMForceFieldOptim", Moved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },

    // SofaMiscFem was deprecated in #2759
    { "FastTetrahedralCorotationalForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "StandardTetrahedralFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangleFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularAnisotropicFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "QuadBendingFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "BoyceAndArruda", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "Costa", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "HyperelasticMaterial", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "MooneyRivlin", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "NeoHookean", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "Ogden", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "PlasticMaterial", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "StandardTetrahedralFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "STVenantKirchhoff", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "TetrahedronHyperelasticityFEMForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "VerondaWestman", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "TetrahedralTensorMassForceField", Moved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.TensorMass") },

    // SofaDeformable was deprecated in #2759
    { "AngularSpringForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "MeshSpringForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "RestShapeSpringsForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "PolynomialRestShapeSpringsForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "SpringForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "StiffSpringForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "PolynomialSpringsForceField", Moved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },

    // SofaGeneralDeformable was deprecated in #2759
    { "FastTriangularBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "FrameSpringForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "QuadBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "QuadularBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "RegularGridSpringForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangleBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangleBendingSprings", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularBiquadraticSpringsForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularQuadraticSpringsForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "VectorSpringForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularTensorMassForceField", Moved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.TensorMass") },

    // SofaGeneralObjectInteraction was deprecated in #2759 and #3039
    { "RepulsiveSpringForceField", Moved("v22.06", "SofaGeneralObjectInteraction", "Sofa.Component.SolidMechanics.Spring") },
    { "InteractionEllipsoidForceField", Moved("v22.06", "SofaGeneralObjectInteraction", "Sofa.Component.MechanicalLoad") },

    // SofaGeneralObjectInteraction was deprecated in #2790 and ...
    { "AttachConstraint", Moved("v22.06", "SofaGeneralObjectInteraction", "Sofa.Component.Constraint.Projective") },

    // SofaBoundaryCondition was deprecated in #2790 and #2759
    { "AffineMovementConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedPlaneConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedRotationConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedTranslationConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "HermiteSplineConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "LinearMovementConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "LinearVelocityConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "OscillatorConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ParabolicConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PartialFixedConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PartialLinearMovementConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PatchTestMovementConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PointConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PositionBasedDynamicsConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectDirectionConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToLineConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToPlaneConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToPointConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "AttachConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "SkeletalMotionConstraint", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ConicalForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "ConstantForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "DiagonalVelocityDampingForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "EdgePressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "EllipsoidForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "LinearForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "OscillatingTorsionPressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "PlaneForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "QuadPressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "SphereForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "SurfacePressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "TaitSurfacePressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "TorsionForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "TrianglePressureForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },
    { "UniformVelocityDampingForceField", Moved("v22.06", "SofaBoundaryCondition", "Sofa.Component.MechanicalLoad") },

    // SofaBaseCollision was deprecated in #2813
    { "BruteForceBroadPhase", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "BruteForceDetection", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "BVHNarrowPhase", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "DefaultPipeline", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "DiscreteIntersection", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Intersection") },
    { "MinProximityIntersection", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Intersection") },
    { "NewProximityIntersection", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Detection.Intersection") },
    { "CubeCollisionModel", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Geometry") },
    { "SphereCollisionModel", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Geometry") },
    { "CylinderCollisionModel", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Geometry") },
    { "DefaultContactManager", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Response.Contact") },
    { "ContactListener", Moved("v22.06", "SofaBaseCollision", "Sofa.Component.Collision.Response.Contact") },

    // SofaMeshCollision was deprecated in #2813
    { "PointCollisionModel", Moved("v22.06", "SofaMeshCollision", "Sofa.Component.Collision.Geometry") },
    { "LineCollisionModel", Moved("v22.06", "SofaMeshCollision", "Sofa.Component.Collision.Geometry") },
    { "TriangleCollisionModel", Moved("v22.06", "SofaMeshCollision", "Sofa.Component.Collision.Geometry") },

    // SofaGeneralMeshCollision was deprecated in #2813
    { "DirectSAP", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "DirectSAPNarrowPhase", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "IncrSAPClassSofaVector", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "RayTraceNarrowPhase", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "RayTraceDetection", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Detection.Algorithm") },
    { "TriangleOctreeModel", Moved("v22.06", "SofaGeneralMeshCollision", "Sofa.Component.Collision.Geometry") },

    // SofaUserInteraction was deprecated in #2813
    { "RayCollisionModel", Moved("v22.06", "SofaUserInteraction", "Sofa.Component.Collision.Geometry") },
    { "Controller", Moved("v22.06", "SofaUserInteraction", "Sofa.Component.Controller") },
    { "MechanicalStateController", Moved("v22.06", "SofaUserInteraction", "Sofa.Component.Controller") },

    // SofaObjectInteraction was deprecated in #2813
    { "PenalityContactForceField", Moved("v22.06", "SofaObjectInteraction", "Sofa.Component.Collision.Response.Contact") },

    // SofaEngine was deprecated in #2812
    { "BoxROI", Moved("v22.06", "SofaEngine", "Sofa.Component.Engine.Select") },

    // SofaGeneralEngine was deprecated in #2812
    { "AverageCoord", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "BoxROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "ClusteringEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "ComplementaryROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "DifferenceEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "DilateEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "DisplacementTransformEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "ExtrudeEdgesAndGenerateQuads", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "ExtrudeQuadsAndGenerateHexas", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "ExtrudeSurface", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "GenerateCylinder", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "GenerateGrid", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "GenerateRigidMass", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "GenerateSphere", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "GroupFilterYoungModulus", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "HausdorffDistance", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "IndexValueMapper", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "Indices2ValuesMapper", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "IndicesFromValues", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "InvertTransformMatrixEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "JoinPoints", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MapIndices", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "MathOp", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "MergeMeshes", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MergePoints", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MergeROIs", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "MergeSets", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MergeVectors", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MeshBarycentricMapperEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MeshBoundaryROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "MeshClosingEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "MeshROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "MeshSampler", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "MeshSplittingEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "MeshSubsetEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "NearestPointROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "NormEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "NormalsFromPoints", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "PairBoxROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "PlaneROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "PointsFromIndices", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "ProximityROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "QuatToRigidEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "ROIValueMapper", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "RandomPointDistributionInSurface", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "RigidToQuatEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "RotateTransformMatrixEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "ScaleTransformMatrixEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "SelectConnectedLabelsROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "SelectLabelROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "ShapeMatching", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "SmoothMeshEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "SphereROI", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "Spiral", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Generate") },
    { "SubsetTopology", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "SumEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "TransformEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "TransformPosition", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "TranslateTransformMatrixEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "ValuesFromIndices", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "ValuesFromPositions", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Select") },
    { "Vertex2Frame", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },

    // SofaMiscEngine was deprecated in #2812
    { "Distances", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Analyze") },
    { "DisplacementMatrixEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },
    { "ProjectiveTransformEngine", Moved("v22.06", "SofaGeneralEngine", "Sofa.Component.Engine.Transform") },

    // SofaMiscExtra was deprecated in #2917
    { "MeshTetraStuffing", Moved("v22.06", "SofaMiscExtra", "Sofa.Component.Engine.Generate") },

    // SofaMiscCollision was deprecated in #2813 and #2820
    { "OBBModel", Moved("v22.06", "SofaMiscCollision", "CollisionOBBCapsule") },
    { "RigidCapsuleCollisionModel", Moved("v22.06", "SofaMiscCollision", "CollisionOBBCapsule") },
    { "CapsuleCollisionModel", Moved("v22.06", "SofaMiscCollision", "CollisionOBBCapsule") },
    { "TriangleModelInRegularGrid", Moved("v22.06", "SofaMiscCollision", "Sofa.Component.Collsion.Geometry") },
    { "TetrahedronCollisionModel", Moved("v22.06", "SofaMiscCollision", "Sofa.Component.Collsion.Geometry") },
    { "RuleBasedContactManager", Moved("v22.06", "SofaMiscCollision", "Sofa.Component.Collsion.Response.Contact") },

    // SofaHaptics was deprecated in #3039
    { "ForceFeedback", Moved("v22.06", "SofaHaptics", "Sofa.Component.Haptics") },
    { "LCPForceFeedback", Moved("v22.06", "SofaHaptics", "Sofa.Component.Haptics") },
    { "MechanicalStateForceFeedback", Moved("v22.06", "SofaHaptics", "Sofa.Component.Haptics") },
    { "NullForceFeedback", Moved("v22.06", "SofaHaptics", "Sofa.Component.Haptics") },

    // SofaValidation was deprecated in #3039
    { "CompareState", Moved("v22.06", "SofaValidation", "Sofa.Component.Playback") },
    { "CompareTopology", Moved("v22.06", "SofaValidation", "Sofa.Component.Playback") },

    // Removed in #4040, deprecated in #2777
    { "MechanicalMatrixMapper", Removed("v23.06", "v23.12") },
    { "MappingGeometricStiffnessForceField", Removed("v23.06", "v23.12") },

    // Moved to CSparseSolvers
    { "SparseCholeskySolver", Moved("v23.12", "Sofa.Component.LinearSolver.Direct", "CSparseSolvers") },
    { "SparseLUSolver", Moved("v23.12", "Sofa.Component.LinearSolver.Direct", "CSparseSolvers") }

};

const std::map<std::string, ComponentChange, std::less<> > uncreatableComponents = {

    /***********************/
    // REMOVED SINCE v23.06

    { "OglGrid", Removed("v22.12", "v23.06")},
    { "OglLineAxis", Removed("v22.12", "v23.06")},

    /***********************/
    // REMOVED SINCE v22.06

    {"PointConstraint", Removed("v21.12", "v22.06")},

    /***********************/
    // REMOVED SINCE v21.12

    { "LMDNewProximityIntersection", Removed("v21.12", "v21.12") },
    { "LocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "LineLocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "PointLocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "TriangleLocalMinDistanceFilter", Removed("v21.12", "v21.12") },

    /***********************/
    // REMOVED SINCE v21.06

    {"LengthContainer", Removed("v21.06", "v21.06")},
    {"PoissonContainer", Removed("v21.06", "v21.06")},
    {"RadiusContainer", Removed("v21.06", "v21.06")},
    {"StiffnessContainer", Removed("v21.06", "v21.06")},

    /***********************/
    // REMOVED SINCE v20.12

    { "DynamicSparseGridTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "HexahedronSetTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "TetrahedronSetTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "QuadSetTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "TriangleSetTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "EdgeSetTopologyAlgorithms", Removed("v20.12", "v20.12") },
    { "PointSetTopologyAlgorithms", Removed("v20.12", "v20.12") },

    /***********************/
    // REMOVED SINCE v20.06

    {"Euler", Removed("v19.12", "v20.06")},
    {"EulerExplicit", Removed("v19.12", "v20.06")},
    {"ExplicitEuler", Removed("v19.12", "v20.06")},
    {"EulerSolver", Removed("v19.12", "v20.06")},
    {"ExplicitEulerSolver", Removed("v19.12", "v20.06")},

    {"Capsule", Removed("v19.12", "v20.06")},
    {"CapsuleModel", Removed("v19.12", "v20.06")},
    {"TCapsuleModel", Removed("v19.12", "v20.06")},

    {"Cube", Removed("v19.12", "v20.06")},
    {"CubeModel", Removed("v19.12", "v20.06")},

    {"CudaPoint", Removed("v19.12", "v20.06")},
    {"CudaPointModel", Removed("v19.12", "v20.06")},

    {"Cylinder", Removed("v19.12", "v20.06")},
    {"CylinderModel", Removed("v19.12", "v20.06")},

    {"Line", Removed("v19.12", "v20.06")},
    {"TLineModel", Removed("v19.12", "v20.06")},
    {"LineMeshModel", Removed("v19.12", "v20.06")},
    {"LineSetModel", Removed("v19.12", "v20.06")},
    {"LineMesh", Removed("v19.12", "v20.06")},
    {"LineSet", Removed("v19.12", "v20.06")},
    {"LineModel", Removed("v19.12", "v20.06")},

    {"OBB", Removed("v19.12", "v20.06")},
    {"OBBModel", Removed("v19.12", "v20.06")},
    {"TOBBModel", Removed("v19.12", "v20.06")},

    {"Point", Removed("v19.12", "v20.06")},
    {"TPointModel", Removed("v19.12", "v20.06")},
    {"PointModel", Removed("v19.12", "v20.06")},
    {"PointMesh", Removed("v19.12", "v20.06")},
    {"PointSet", Removed("v19.12", "v20.06")},

    {"Ray", Removed("v19.12", "v20.06")},
    {"RayModel", Removed("v19.12", "v20.06")},

    {"RigidCapsule", Removed("v19.12", "v20.06")},
    {"RigidCapsuleModel", Removed("v19.12", "v20.06")},
    {"RigidCapsuleCollisionModel", Removed("v19.12", "v20.06")},

    {"Sphere", Removed("v19.12", "v20.06")},
    {"SphereModel", Removed("v19.12", "v20.06")},
    {"TSphereModel", Removed("v19.12", "v20.06")},

    {"Tetrahedron", Removed("v19.12", "v20.06")},
    {"TetrahedronModel", Removed("v19.12", "v20.06")},

    {"Triangle", Removed("v19.12", "v20.06")},
    {"TriangleSet", Removed("v19.12", "v20.06")},
    {"TriangleMesh", Removed("v19.12", "v20.06")},
    {"TriangleSetModel", Removed("v19.12", "v20.06")},
    {"TriangleMeshModel", Removed("v19.12", "v20.06")},
    {"TriangleModel", Removed("v19.12", "v20.06")},
    {"TTriangleModel", Removed("v19.12", "v20.06")},

};


const std::map< std::string, Renamed, std::less<> > renamedComponents = {
    // Change Constraint naming #4302
    {"AffineMovementConstraint", Renamed("v24.06","v25.06","AffineMovementProjectiveConstraint")},
    {"AttachConstraint", Renamed("v24.06","v25.06","AttachProjectiveConstraint")},
    {"FixedConstraint", Renamed("v24.06","v25.06","FixedProjectiveConstraint")},
    {"FixedPlaneConstraint", Renamed("v24.06","v25.06","FixedPlaneProjectiveConstraint")},
    {"FixedRotationConstraint", Renamed("v24.06","v25.06","FixedRotationProjectiveConstraint")},
    {"FixedTranslationConstraint", Renamed("v24.06","v25.06","FixedTranslationProjectiveConstraint")},
    {"HermiteSplineConstraint", Renamed("v24.06","v25.06","HermiteSplineProjectiveConstraint")},
    {"LinearMovementConstraint", Renamed("v24.06","v25.06","LinearMovementProjectiveConstraint")},
    {"LinearVelocityConstraint", Renamed("v24.06","v25.06","LinearVelocityProjectiveConstraint")},
    {"OscillatorConstraint", Renamed("v24.06","v25.06","OscillatorProjectiveConstraint")},
    {"ParabolicConstraint", Renamed("v24.06","v25.06","ParabolicProjectiveConstraint")},
    {"PartialFixedConstraint", Renamed("v24.06","v25.06","PartialFixedProjectiveConstraint")},
    {"PartialLinearMovementConstraint", Renamed("v24.06","v25.06","PartialLinearMovementProjectiveConstraint")},
    {"PatchTestMovementConstraint", Renamed("v24.06","v25.06","PatchTestMovementProjectiveConstraint")},
    {"PositionBasedDynamicsConstraint", Renamed("v24.06","v25.06","PositionBasedDynamicsProjectiveConstraint")},
    {"SkeletalMotionConstraint", Renamed("v24.06","v25.06","SkeletalMotionProjectiveConstraint")},
    {"ProjectToLineConstraint", Renamed("v24.06","v25.06","LineProjectiveConstraint")},
    {"ProjectToPlaneConstraint", Renamed("v24.06","v25.06","PlaneProjectiveConstraint")},
    {"ProjectToPointConstraint", Renamed("v24.06","v25.06","PointProjectiveConstraint")},
    {"ProjectDirectionConstraint", Renamed("v24.06","v25.06","DirectionProjectiveConstraint")},
    {"BilateralInteractionConstraint", Renamed("v24.06","v25.06","BilateralLagrangianConstraint")},
    {"SlidingConstraint", Renamed("v24.06","v25.06","SlidingLagrangianConstraint")},
    {"StopperConstraint", Renamed("v24.06","v25.06","StopperLagrangianConstraint")},
    {"UniformConstraint", Renamed("v24.06","v25.06","UniformLagrangianConstraint")},
    {"UnilateralInteractionConstraint", Renamed("v24.06","v25.06","UnilateralLagrangianConstraint")}

};

} // namespace sofa::helper::lifecycle
