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
#include "ComponentChange.h"


namespace sofa::helper::lifecycle
{

const std::map<std::string, Deprecated, std::less<> > deprecatedComponents = {
    // SofaMiscForceField
    {"MatrixMass", Deprecated("v19.06", "v19.12")},
    {"RayTraceDetection", Deprecated("v21.06", "v21.12")},
    {"BruteForceDetection", Deprecated("v21.06", "v21.12")},
    {"DirectSAP", Deprecated("v21.06", "v21.12")},
    {"PointConstraint", Deprecated("v21.12", "v22.06")},
};

const std::map<std::string, ComponentChange, std::less<> > uncreatableComponents = {
    // SofaDistanceGrid was pluginized in #389
    {"BarycentricPenalityContact", Pluginized("v17.12", "SofaMeshCollision")},
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

    // SofaDenseSolver was pluginized in #1299
    {"LULinearSolver", Pluginized("v20.06", "SofaDenseSolver")},
    //{"NewMatCholeskySolver", Pluginized("v20.06", "SofaDenseSolver")},
    //{"NewMatCGLinearSolver", Pluginized("v20.06", "SofaDenseSolver")},

    // SofaNonUniformFem was pluginized in #1344
    {"DynamicSparseGridGeometryAlgorithms", Pluginized("v20.06", "SofaNonUniformFem")},
    {"DynamicSparseGridTopologyAlgorithms", Pluginized("v20.06", "SofaNonUniformFem")},
    {"DynamicSparseGridTopologyContainer", Pluginized("v20.06", "SofaNonUniformFem")},
    {"DynamicSparseGridTopologyModifier", Pluginized("v20.06", "SofaNonUniformFem")},
    {"HexahedronCompositeFEMForceFieldAndMass", Pluginized("v20.06", "SofaNonUniformFem")},
    {"HexahedronCompositeFEMMapping", Pluginized("v20.06", "SofaNonUniformFem")},
    {"MultilevelHexahedronSetTopologyContainer", Pluginized("v20.06", "SofaNonUniformFem")},
    {"NonUniformHexahedralFEMForceFieldAndMass", Pluginized("v20.06", "SofaNonUniformFem")},
    {"NonUniformHexahedronFEMForceFieldAndMass", Pluginized("v20.06", "SofaNonUniformFem")},
    {"SparseGridMultipleTopology", Pluginized("v20.06", "SofaNonUniformFem")},
    {"SparseGridRamificationTopology", Pluginized("v20.06", "SofaNonUniformFem")},

    // SofaMiscEngine was pluginized in #1520
    { "DisplacementTransformEngine", Pluginized("v20.12", "SofaMiscEngine") },
    { "Distances", Pluginized("v20.12", "SofaMiscEngine") },
    { "ProjectiveTransformEngine", Pluginized("v20.12", "SofaMiscEngine") },

    // SofaMiscExtra was pluginized in #1520
    { "MeshTetraStuffing", Pluginized("v20.12", "SofaMiscExtra") },

    // SofaMiscFem was pluginized in #1520
    { "FastTetrahedralCorotationalForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "StandardTetrahedralFEMForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "TetrahedralTensorMassForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "TetrahedronHyperelasticityFEMForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "TriangleFEMForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "TriangularAnisotropicFEMForceField", Pluginized("v20.12", "SofaMiscFem") },
    { "TriangularFEMForceField", Pluginized("v20.12", "SofaMiscFem") },
        
    // SofaMiscForceField was pluginized in #1520
    { "GearSpringForceField", Pluginized("v20.12", "SofaMiscForceField") },
    { "MeshMatrixMass", Pluginized("v20.12", "SofaMiscForceField") },

    // SofaMiscMapping was pluginized in #1520
    { "BeamLinearMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "CenterOfMassMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "CenterOfMassMulti2Mapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "CenterOfMassMultiMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "DeformableOnRigidFrameMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "DistanceFromTargetMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "DistanceMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "IdentityMultiMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "SquareDistanceMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "SquareMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "SubsetMultiMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "TubularMapping", Pluginized("v20.12", "SofaMiscMapping") },
    { "VoidMapping", Pluginized("v20.12", "SofaMiscMapping") },

    // SofaMiscSolver was pluginized in #1520
    { "DampVelocitySolver", Pluginized("v20.12", "SofaMiscSolver") },
    { "NewmarkImplicitSolver", Pluginized("v20.12", "SofaMiscSolver") },

    // SofaMiscTopology was pluginized in #1520
    { "TopologicalChangeProcessor", Pluginized("v20.12", "SofaMiscTopology") },

    // SofaGeneralVisual was pluginized in #1530
    { "RecordedCamera", Pluginized("v20.12", "SofaGeneralVisual") },
    { "VisualTransform", Pluginized("v20.12", "SofaGeneralVisual") },
    { "Visual3DText", Pluginized("v20.12", "SofaGeneralVisual") },
        
    // SofaGraphComponent was pluginized in #1531
    { "AddFrameButtonSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "AddRecordedCameraButtonSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "AttachBodyButtonSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "FixPickedParticleButtonSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "Gravity", Pluginized("v20.12", "SofaGraphComponent") },
    { "InteractingBehaviorModel", Pluginized("v20.12", "SofaGraphComponent") },
    { "MouseButtonSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "PauseAnimation", Pluginized("v20.12", "SofaGraphComponent") },
    { "PauseAnimationOnEvent", Pluginized("v20.12", "SofaGraphComponent") },
    { "SofaDefaultPathSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "StatsSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "ViewerSetting", Pluginized("v20.12", "SofaGraphComponent") },
    { "APIVersion", Pluginized("v20.12", "SofaGraphComponent") },

    // SofaBoundaryCondition was pluginized in #1556
    { "AffineMovementConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ConicalForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ConstantForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "DiagonalVelocityDampingForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "EdgePressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "EllipsoidForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "FixedConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "FixedPlaneConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "FixedRotationConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "FixedTranslationConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "HermiteSplineConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "LinearForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "LinearMovementConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "LinearVelocityConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "OscillatingTorsionPressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "OscillatorConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ParabolicConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PartialFixedConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PartialLinearMovementConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PatchTestMovementConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PlaneForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PointConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "PositionBasedDynamicsConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "QuadPressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "SkeletalMotionConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "SphereForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "SurfacePressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "TaitSurfacePressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "TorsionForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "TrianglePressureForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "UniformVelocityDampingForceField", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ProjectToLineConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ProjectToPlaneConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ProjectToPointConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },
    { "ProjectDirectionConstraint", Pluginized("v20.12", "SofaBoundaryCondition") },

    // SofaGeneralAnimationLoop was pluginized in #1563
    { "MechanicalMatrixMapper", Pluginized("v20.12", "SofaGeneralAnimationLoop") },
    { "MultiStepAnimationLoop", Pluginized("v20.12", "SofaGeneralAnimationLoop") },
    { "MultiTagAnimationLoop", Pluginized("v20.12", "SofaGeneralAnimationLoop") },

    // SofaGeneralDeformable was pluginized in #1564
    { "FastTriangularBendingSprings", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "FrameSpringForceField", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "QuadBendingSprings", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "RegularGridSpringForceField", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "QuadularBendingSprings", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "TriangleBendingSprings", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "TriangularBendingSprings", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "TriangularBiquadraticSpringsForceField", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "TriangularQuadraticSpringsForceField", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "TriangularTensorMassForceField", Pluginized("v20.12", "SofaGeneralDeformable") },
    { "VectorSpringForceField", Pluginized("v20.12", "SofaGeneralDeformable") },

    // SofaGeneralEngine was pluginized in #1565
    { "AverageCoord", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ClusteringEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ComplementaryROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "DilateEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "DifferenceEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ExtrudeEdgesAndGenerateQuads", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ExtrudeQuadsAndGenerateHexas", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ExtrudeSurface", Pluginized("v20.12", "SofaGeneralEngine") },
    { "GenerateCylinder", Pluginized("v20.12", "SofaGeneralEngine") },
    { "GenerateGrid", Pluginized("v20.12", "SofaGeneralEngine") },
    { "GenerateRigidMass", Pluginized("v20.12", "SofaGeneralEngine") },
    { "GenerateSphere", Pluginized("v20.12", "SofaGeneralEngine") },
    { "GroupFilterYoungModulus", Pluginized("v20.12", "SofaGeneralEngine") },
    { "HausdorffDistance", Pluginized("v20.12", "SofaGeneralEngine") },
    { "IndexValueMapper", Pluginized("v20.12", "SofaGeneralEngine") },
    { "Indices2ValuesMapper", Pluginized("v20.12", "SofaGeneralEngine") },
    { "IndicesFromValues", Pluginized("v20.12", "SofaGeneralEngine") },
    { "JoinPoints", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MapIndices", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MathOp", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MergeMeshes", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MergePoints", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MergeROIs", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MergeSets", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MergeVectors", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshBarycentricMapperEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshClosingEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshBoundaryROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshSampler", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshSplittingEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "MeshSubsetEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "NearestPointROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "NormEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "NormalsFromPoints", Pluginized("v20.12", "SofaGeneralEngine") },
    { "PairBoxRoi", Pluginized("v20.12", "SofaGeneralEngine") },
    { "PlaneROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "PointsFromIndices", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ProximityROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "QuatToRigidEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ROIValueMapper", Pluginized("v20.12", "SofaGeneralEngine") },
    { "RandomPointDistributionInSurface", Pluginized("v20.12", "SofaGeneralEngine") },
    { "RigidToQuatEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SelectLabelROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SelectConnectedLabelsROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ShapeMatching", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SmoothMeshEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SphereROI", Pluginized("v20.12", "SofaGeneralEngine") },
    { "Spiral", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SubsetTopology", Pluginized("v20.12", "SofaGeneralEngine") },
    { "SumEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "TextureInterpolation", Pluginized("v20.12", "SofaGeneralEngine") },
    { "TransformEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "TransformMatrixEngine", Pluginized("v20.12", "SofaGeneralEngine") },
    { "TransformPosition", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ValuesFromIndices", Pluginized("v20.12", "SofaGeneralEngine") },
    { "ValuesFromPositions", Pluginized("v20.12", "SofaGeneralEngine") },
    { "Vertex2Frame", Pluginized("v20.12", "SofaGeneralEngine") },

    // SofaGeneralExplicitOdeSolver was pluginized in #1566
    { "CentralDifferenceSolver", Pluginized("v20.12", "SofaGeneralExplicitOdeSolver") },
    { "RungeKutta2Solver", Pluginized("v20.12", "SofaGeneralExplicitOdeSolver") },
    { "RungeKutta4Solver", Pluginized("v20.12", "SofaGeneralExplicitOdeSolver") },

    // SofaGeneralImplicitOdeSolver was pluginized in #1572
    { "VariationalSymplecticSolver", Pluginized("v20.12", "SofaGeneralImplicitOdeSolver") },

    // SofaGeneralLinearSolver was pluginized in #1575
    { "BTDLinearSolver", Pluginized("v20.12", "SofaGeneralLinearSolver") },
    { "CholeskySolver", Pluginized("v20.12", "SofaGeneralLinearSolver") },
    { "MinResLinearSolver", Pluginized("v20.12", "SofaGeneralLinearSolver") },

    // SofaGeneralRigid was pluginized in #1579
    { "ArticulatedHierarchyContainer", Pluginized("v20.12", "SofaGeneralRigid") },
    { "ArticulationCenter", Pluginized("v20.12", "SofaGeneralRigid") },
    { "Articulation", Pluginized("v20.12", "SofaGeneralRigid") },
    { "ArticulatedSystemMapping", Pluginized("v20.12", "SofaGeneralRigid") },
    { "LineSetSkinningMapping", Pluginized("v20.12", "SofaGeneralRigid") },
    { "SkinningMapping", Pluginized("v20.12", "SofaGeneralRigid") },

    // SofaGeneralObjectInteraction was pluginized in #1580
    { "AttachConstraint", Pluginized("v20.12", "SofaGeneralObjectInteraction") },
    { "BoxStiffSpringForceField", Pluginized("v20.12", "SofaGeneralObjectInteraction") },
    { "InteractionEllipsoidForceField", Pluginized("v20.12", "SofaGeneralObjectInteraction") },
    { "RepulsiveSpringForceField", Pluginized("v20.12", "SofaGeneralObjectInteraction") },

    // SofaGeneralSimpleFem was pluginized in #1582
    { "BeamFEMForceField", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "HexahedralFEMForceField", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "HexahedralFEMForceFieldAndMass", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "TetrahedralCorotationalFEMForceField", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "TriangularFEMForceFieldOptim", Pluginized("v20.12", "SofaGeneralSimpleFem") },

    // SofaGeneralTopology was pluginized in #1583
    { "CubeTopology", Pluginized("v20.12", "SofaGeneralTopology") },
    { "CylinderGridTopology", Pluginized("v20.12", "SofaGeneralTopology") },
    { "SphereGridTopology", Pluginized("v20.12", "SofaGeneralTopology") },
    { "SphereQuadTopology", Pluginized("v20.12", "SofaGeneralTopology") },

    // SofaGeneralTopology was pluginized in #1586
    { "CenterPointTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Edge2QuadTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Hexa2QuadTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Hexa2TetraTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "IdentityTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Mesh2PointMechanicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Mesh2PointTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Quad2TriangleTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "SimpleTesselatedHexaTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "SimpleTesselatedTetraMechanicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "SimpleTesselatedTetraTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "SubsetTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Tetra2TriangleTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },
    { "Triangle2EdgeTopologicalMapping", Pluginized("v20.12", "SofaTopologyMapping") },

    // SofaUserInteraction was pluginized in #1588
    { "MechanicalStateController", Pluginized("v20.12", "SofaUserInteraction") },
    { "MouseInteractor", Pluginized("v20.12", "SofaUserInteraction") },
    { "RayModel", Pluginized("v20.12", "SofaUserInteraction") },
    { "RayTraceDetection", Pluginized("v20.12", "SofaUserInteraction") },
    { "SleepController", Pluginized("v20.12", "SofaUserInteraction") },

    // SofaConstraint was pluginized in #1592
    { "BilateralInteractionConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "ConstraintAnimationLoop", Pluginized("v20.12", "SofaConstraint") },
    { "FreeMotionAnimationLoop", Pluginized("v20.12", "SofaConstraint") },
    { "GenericConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "GenericConstraintSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LCPConstraintSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LinearSolverConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "LocalMinDistance", Pluginized("v20.12", "SofaConstraint") },
    { "MappingGeometricStiffnessForceField", Pluginized("v20.12", "SofaConstraint") },
    { "PrecomputedConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "SlidingConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "StopperConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "UncoupledConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "UniformConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "UnilateralInteractionConstraint", Pluginized("v20.12", "SofaConstraint") },

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



    // SofaGeneralLoader was pluginized in #1595
    { "GIDMeshLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "GridMeshCreator", Pluginized("v20.12", "SofaGeneralLoader") },
    { "InputEventReader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "MeshGmshLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "MeshOffLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "MeshSTLLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "MeshTrianLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "MeshXspLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "OffSequenceLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "ReadState", Pluginized("v20.12", "SofaGeneralLoader") },
    { "ReadTopology", Pluginized("v20.12", "SofaGeneralLoader") },
    { "SphereLoader", Pluginized("v20.12", "SofaGeneralLoader") },
    { "StringMeshCreator", Pluginized("v20.12", "SofaGeneralLoader") },
    { "VoxelGridLoader", Pluginized("v20.12", "SofaGeneralLoader") },

     // SofaSimpleFem was pluginized in #1598
    { "HexahedronFEMForceField", Pluginized("v20.12", "SofaSimpleFem") },
    { "TetrahedronDiffusionFEMForceField", Pluginized("v20.12", "SofaSimpleFem") },
    { "TetrahedronFEMForceField", Pluginized("v20.12", "SofaSimpleFem") },

    // SofaRigid was pluginized in #1599
    { "JointSpringForceField", Pluginized("v20.12", "SofaRigid") },
    { "RigidMapping", Pluginized("v20.12", "SofaRigid") },
    { "RigidRigidMapping", Pluginized("v20.12", "SofaRigid") },

    // SofaDeformable was pluginized in #1600
    { "AngularSpringForceField", Pluginized("v20.12", "SofaDeformable") },
    { "MeshSpringForceField", Pluginized("v20.12", "SofaDeformable") },
    { "PolynomialRestShapeSpringsForceField", Pluginized("v20.12", "SofaDeformable") },
    { "PolynomialSpringsForceField", Pluginized("v20.12", "SofaDeformable") },
    { "RestShapeSpringsForceField", Pluginized("v20.12", "SofaDeformable") },
    { "SpringForceField", Pluginized("v20.12", "SofaDeformable") },
    { "StiffSpringForceField", Pluginized("v20.12", "SofaDeformable") },

    // SofaObjectInteraction was pluginized in #1601
    { "PenalityContactForceField", Pluginized("v20.12", "SofaObjectInteraction") },

    // SofaMeshCollision was pluginized in #1602
    { "LineCollisionModel", Pluginized("v20.12", "SofaMeshCollision") },
    { "PointCollisionModel", Pluginized("v20.12", "SofaMeshCollision") },
    { "TriangleCollisionModel", Pluginized("v20.12", "SofaMeshCollision") },

    // SofaEngine was pluginized in #1603
    { "BoxROI", Pluginized("v20.12", "SofaEngine") },

    // SofaExplicitOdeSolver was pluginized in #1606
    { "EulerExplicitSolver", Pluginized("v20.12", "SofaExplicitOdeSolver") },

    // SofaImplicitOdeSolver was pluginized in #1607
    { "EulerImplicitSolver", Pluginized("v20.12", "SofaImplicitOdeSolver") },
    { "StaticSolver", Pluginized("v20.12", "SofaImplicitOdeSolver") },

    // SofaLoader was pluginized in #1608
    { "MeshOBJLoader", Pluginized("v20.12", "SofaLoader") },
    { "MeshVTKLoader", Pluginized("v20.12", "SofaLoader") },

    // SofaEigen2Solver was pluginized in #1635
    { "SVDLinearSolver", Pluginized("v20.12", "SofaEigen2Solver") },

    // SofaBaseUtils was packaged in #1640
    //{ "AddResourceRepository", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "AddPluginRepository", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "InfoComponent", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "MakeAliasComponent", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "MakeDataAliasComponent", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "MessageHandlerComponent", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "FileMessageHandlerComponent", Pluginized("v20.12", "SofaBaseUtils") },
    //{ "RequiredPlugin", Pluginized("v20.12", "SofaBaseUtils") },

    // SofaBaseCollision was packaged in #1653
    //{ "BruteForceDetection", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "CapsuleCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "ContactListener", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "CubeCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "CylinderCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "DefaultContactManager", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "DefaultPipeline", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "DiscreteIntersection", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "MinProximityIntersection", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "NewProximityIntersection", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "OBBCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "RigidCapsuleCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },
    //{ "SphereCollisionModel", Pluginized("v20.12", "SofaBaseCollision") },

    // SofaBaseLinearSolver was packaged in #1655
    //{ "CGLinearSolver", Pluginized("v20.12", "SofaBaseLinearSolver") },

    // SofaBaseTopology was packaged in #1676
    //{ "EdgeSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "EdgeSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "EdgeSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "GridTopology", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "HexahedronSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "HexahedronSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "HexahedronSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "MeshTopology", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "PointSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "PointSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "PointSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "QuadSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "QuadSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "QuadSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "RegularGridTopology", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "SparseGridTopology", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TetrahedronSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TetrahedronSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TetrahedronSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TriangleSetGeometryAlgorithms", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TriangleSetTopologyContainer", Pluginized("v20.12", "SofaBaseTopology") },
    //{ "TriangleSetTopologyModifier", Pluginized("v20.12", "SofaBaseTopology") },

    // SofaBaseVisual was packaged in #1677
    //{ "BackgroundSetting", Pluginized("v20.12", "SofaBaseVisual") },
    //{ "Camera", Pluginized("v20.12", "SofaBaseVisual") },
    //{ "InteractiveCamera", Pluginized("v20.12", "SofaBaseVisual") },
    //{ "VisualModelImpl", Pluginized("v20.12", "SofaBaseVisual") },
    //{ "VisualStyle", Pluginized("v20.12", "SofaBaseVisual") },

    // SofaBaseMechanics was packaged in #1680
    //{ "BarycentricMapping", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "DiagonalMass", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "IdentityMapping", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "MappedObject", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "MechanicalObject", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "SubsetMapping", Pluginized("v20.12", "SofaBaseMechanics") },
    //{ "UniformMass", Pluginized("v20.12", "SofaBaseMechanics") },

    /***********************/
    // REMOVED SINCE v21.12

    { "LMDNewProximityIntersection", Removed("v21.12", "v21.12") },
    { "LocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "LineLocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "PointLocalMinDistanceFilter", Removed("v21.12", "v21.12") },
    { "TriangleLocalMinDistanceFilter", Removed("v21.12", "v21.12") },

    /***********************/
    // REMOVED SINCE v21.06

    {"LennardJonesForceField", Removed("v17.12", "v21.06")},
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

    /***********************/
    // MOVED SINCE v21.06
    { "OBBCollisionModel", Moved("v21.06", "SofaBaseCollision", "SofaMiscCollision") },
    { "RigidCapsuleCollisionModel", Moved("v21.06", "SofaBaseCollision", "SofaMiscCollision") },
    { "CapsuleModel", Moved("v21.06", "SofaBaseCollision", "SofaMiscCollision") },
    { "SpatialGridPointModel", Moved("v21.06", "SofaMiscCollision", "SofaSphFluid") },

    // MOVED SINCE v21.12
    { "LULinearSolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },
    //{"NewMatCholeskySolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },
    //{"NewMatCGLinearSolver", Moved("v21.12", "SofaDenseSolver", "SofaNewmat") },

    // MOVED SINCE v22.06
    { "GlobalSystemMatrixExporter", Moved("v22.06", "SofaBaseLinearSolver", "SofaMatrix") },

};

const std::map< std::string, CreatableMoved, std::less<> > movedComponents = 
{

    /***********************/
    // MOVED SINCE v22.06
    // SofaMiscSolver was deprecated in #2571
    { "DampVelocitySolver", CreatableMoved("v22.06", "SofaMiscSolver", "Sofa.Component.ODESolver.Forward") },
    { "NewmarkImplicitSolver", CreatableMoved("v22.06", "SofaMiscSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaExplicitOdeSolver was deprecated in #2571
    { "EulerExplicitSolver", CreatableMoved("v22.06", "SofaExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },

    // SofaImplicitOdeSolver was deprecated in #2571
    { "EulerImplicitSolver", CreatableMoved("v22.06", "SofaImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },
    { "StaticSolver", CreatableMoved("v22.06", "SofaImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaGeneralExplicitOdeSolver was deprecated in #2571
    { "CentralDifferenceSolver", CreatableMoved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },
    { "RungeKutta2Solver", CreatableMoved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },
    { "RungeKutta4Solver", CreatableMoved("v22.06", "SofaGeneralExplicitOdeSolver", "Sofa.Component.ODESolver.Forward") },

    // SofaGeneralImplicitOdeSolver was deprecated in #2571
    { "VariationalSymplecticSolver", CreatableMoved("v22.06", "SofaGeneralImplicitOdeSolver", "Sofa.Component.ODESolver.Backward") },

    // SofaLoader was deprecated in #2582
    { "MeshOBJLoader", CreatableMoved("v22.06", "SofaLoader", "Sofa.Component.IO.Mesh") },
    { "MeshVTKLoader", CreatableMoved("v22.06", "SofaLoader", "Sofa.Component.IO.Mesh") },

    // SofaGeneralLoader was deprecated in #2582
    { "MeshGmshLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "GIDMeshLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "GridMeshCreator", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshOffLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshSTLLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshTrianLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "MeshXspLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "OffSequenceLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "SphereLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "StringMeshCreator", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "VoxelGridLoader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.IO.Mesh") },
    { "ReadState", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },
    { "ReadTopology", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },
    { "InputEventReader", CreatableMoved("v22.06", "SofaGeneralLoader", "Sofa.Component.Playback") },

    // SofaExporter was deprecated in #2582
    { "BlenderExporter", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "MeshExporter", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "STLExporter", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "VisualModelOBJExporter", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "VTKExporter", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.IO.Mesh") },
    { "WriteState", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.Playback") },
    { "WriteTopology", CreatableMoved("v22.06", "SofaExporter", "Sofa.Component.Playback") },

    // SofaBaseUtils was deprecated in #2582 and ...
    { "AddResourceRepository", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MakeAliasComponent", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MakeDataAliasComponent", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "MessageHandlerComponent", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "FileMessageHandlerComponent", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },
    { "RequiredPlugin", CreatableMoved("v22.06", "SofaBaseUtils", "Sofa.Component.SceneUtility") },

    // SofaGraphComponent was deprecated in #2582 and ...
    { "APIVersion", CreatableMoved("v22.06", "SofaGraphComponent", "Sofa.Component.SceneUtility") },
    
    // SofaBaseTopology was deprecated in #2612
    { "EdgeSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "EdgeSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "HexahedronSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "PointSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "QuadSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TetrahedronSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetGeometryAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyAlgorithms", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyContainer", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "TriangleSetTopologyModifier", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Dynamic") },
    { "MeshTopology", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Constant") },
    { "GridTopology", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },
    { "RegularGridTopology", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },
    { "SparseGridTopology", CreatableMoved("v22.06", "SofaBaseTopology", "Sofa.Component.Topology.Container.Grid") },

    // SofaGeneralTopology was deprecated in #2612
    { "CubeTopology", CreatableMoved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Constant") },
    { "SphereQuadTopology", CreatableMoved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Constant") },
    { "CylinderGridTopology", CreatableMoved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Grid") },
    { "SphereGridTopology", CreatableMoved("v22.06", "SofaGeneralTopology", "Sofa.Component.Topology.Container.Grid") },

    // SofaNonUniformFem was deprecated in #2612 and #2759
    { "DynamicSparseGridGeometryAlgorithms", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyAlgorithms", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyContainer", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "DynamicSparseGridTopologyModifier", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "MultilevelHexahedronSetTopologyContainer", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Dynamic") },
    { "SparseGridMultipleTopology", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Grid") },
    { "SparseGridRamificationTopology", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.Topology.Container.Grid") },
    { "NonUniformHexahedralFEMForceFieldAndMass", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "NonUniformHexahedronFEMForceFieldAndMass", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "HexahedronCompositeFEMForceFieldAndMass", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },
    { "HexahedronCompositeFEMMapping", CreatableMoved("v22.06", "SofaNonUniformFem", "Sofa.Component.SolidMechanics.FEM.NonUniform") },

    // SofaTopologicalMapping was deprecated in #2612 and #XXXX
    { "CenterPointTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Edge2QuadTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Hexa2QuadTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Hexa2TetraTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "IdentityTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Mesh2PointTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Quad2TriangleTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SimpleTesselatedHexaTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SimpleTesselatedTetraTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "SubsetTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Tetra2TriangleTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Triangle2EdgeTopologicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Topology.Mapping") },
    { "Mesh2PointMechanicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Mapping.Linear") },
    { "SimpleTesselatedTetraMechanicalMapping", CreatableMoved("v22.06", "SofaTopologicalMapping", "Sofa.Component.Mapping.Linear") },

    // SofaMiscTopology was deprecated in #2612
    { "TopologicalChangeProcessor", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },
    { "TopologyBoundingTrasher", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },
    { "TopologyChecker", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Topology.Utility") },

    // SofaBaseVisual was deprecated in #2679
    { "Camera", CreatableMoved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "InteractiveCamera", CreatableMoved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "VisualModelImpl", CreatableMoved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },
    { "VisualStyle", CreatableMoved("v22.06", "SofaBaseVisual", "Sofa.Component.Visual") },

    // SofaGeneralVisual was deprecated in #2679
    { "RecordedCamera", CreatableMoved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },
    { "Visual3DText", CreatableMoved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },
    { "VisualTransform", CreatableMoved("v22.06", "SofaGeneralVisual", "Sofa.Component.Visual") },

    // SofaSimpleFem was deprecated in #2753 and ....
    { "TetrahedronDiffusionFEMForceField", CreatableMoved("v22.06", "SofaSimpleFem", "Sofa.Component.Diffusion") },

    // SofaOpenglVisual was deprecated in #2709
    { "DataDisplay", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "MergeVisualModels", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglCylinderModel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglGrid", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglLineAxis", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglModel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "PointSplatModel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "SlicedVolumetricModel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering3D") },
    { "OglColorMap", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "OglLabel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "OglViewport", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Rendering2D") },
    { "ClipPlane", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "CompositingVisualLoop", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "DirectionalLight", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "PositionalLight", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "SpotLight", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "LightManager", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatAttribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntAttribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUIntAttribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt2Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt3Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglUInt4Attribute", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglOITShader", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglRenderingSRGB", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShader", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShaderMacro", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShaderVisualModel", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglShadowShader", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglTexture", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglTexture2D", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglIntVectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt2VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt3VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglInt4VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloatVectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat2VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat3VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglFloat4VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2x3Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3x2Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix2x4Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4x2Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix3x4Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4x3Variable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OglMatrix4VectorVariable", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "OrderIndependentTransparencyManager", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "PostProcessManager", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "VisualManagerPass", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "VisualManagerSecondaryPass", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Shader") },
    { "TextureInterpolation", CreatableMoved("v22.06", "SofaOpenglVisual", "Sofa.GL.Component.Engine") },

    // SofaBaseLinearSolver was deprecated in #2717
    { "CGLinearSolver", CreatableMoved("v22.06", "SofaBaseLinearSolver", "Sofa.Component.LinearSolver.Iterative") },

    // SofaGeneralLinearSolver was deprecated in #2717
    { "MinResLinearSolver", CreatableMoved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Iterative") },
    { "BTDLinearSolver", CreatableMoved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "CholeskySolver", CreatableMoved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },

    // SofaSparseSolver was deprecated in #2717
    { "FillReducingOrdering", CreatableMoved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "PrecomputedLinearSolver", CreatableMoved("v22.06", "SofaGeneralLinearSolver", "Sofa.Component.LinearSolver.Direct") },
    { "SparseCholeskySolver", CreatableMoved("v22.06", "SofaSparseSolver", "Sofa.Component.LinearSolver.Direct") },
    { "SparseLDLSolver", CreatableMoved("v22.06", "SofaSparseSolver", "Sofa.Component.LinearSolver.Direct") },
    { "SparseLUSolver", CreatableMoved("v22.06", "SofaSparseSolver", "Sofa.Component.LinearSolver.Direct") },

    // SofaDenseSolver was deprecated in #2717
    { "SVDLinearSolver", CreatableMoved("v22.06", "SofaDenseSolver", "Sofa.Component.LinearSolver.Direct") },
        
    // SofaPreconditioner was deprecated in #2717
    { "ShewchukPCGLinearSolver", CreatableMoved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Iterative") },
    { "BlockJacobiPreconditioner", CreatableMoved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "PrecomputedWarpPreconditioner", CreatableMoved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "SSORPreconditioner", CreatableMoved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },
    { "WarpPreconditioner", CreatableMoved("v22.06", "SofaPreconditioner", "Sofa.Component.LinearSolver.Preconditioner") },

    // SofaBaseMechanics was deprecated in #2752, #2635 and #2766
    { "DiagonalMass", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mass") },
    { "UniformMass", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mass") },
    { "BarycentricMapping", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "IdentityMapping", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "SubsetMapping", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.Mapping.Linear") },
    { "MechanicalObject", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.StateContainer") },
    { "MappedObject", CreatableMoved("v22.06", "SofaBaseMechanics", "Sofa.Component.StateContainer") },
    
    // SofaMiscForceField was deprecated in #2752 and ...
    { "MeshMatrixMass", CreatableMoved("v22.06", "SofaMiscForceField", "Sofa.Component.Mass") },
    { "GearSpringForceField", CreatableMoved("v22.06", "SofaMiscForceField", "Sofa.Component.SolidMechanics.Spring") },


    // SofaRigid was deprecated in #2635 and #2759
    { "RigidMapping", CreatableMoved("v22.06", "SofaRigid", "Sofa.Component.Mapping.NonLinear") },
    { "RigidRigidMapping", CreatableMoved("v22.06", "SofaRigid", "Sofa.Component.Mapping.NonLinear") },
    { "JointSpringForceField", CreatableMoved("v22.06", "SofaRigid", "Sofa.Component.SolidMechanics.Spring") },

    // SofaGeneralRigid was deprecated in #2635 and ...
    { "LineSetSkinningMapping", CreatableMoved("v22.06", "SofaGeneralRigid", "Sofa.Component.Mapping.Linear") },
    { "SkinningMapping", CreatableMoved("v22.06", "SofaGeneralRigid", "Sofa.Component.Mapping.Linear") },

    // SofaMiscMapping was deprecated in #2635
    { "BeamLinearMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMulti2Mapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "CenterOfMassMultiMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "DeformableOnRigidFrameMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "DistanceFromTargetMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "DistanceMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "IdentityMultiMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "SquareMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "SquareDistanceMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.NonLinear") },
    { "SubsetMultiMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "TubularMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },
    { "VoidMapping", CreatableMoved("v22.06", "SofaMiscMapping", "Sofa.Component.Mapping.Linear") },

    // SofaConstraint was deprecated in #2635, #2790 and #2796
    { "MappingGeometricStiffnessForceField", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Mapping.MappedMatrix") },
    { "BilateralInteractionConstraint", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "GenericConstraintCorrection", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "GenericConstraintSolver", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Solver") },
    { "LCPConstraintSolver", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Solver") },
    { "LinearSolverConstraintCorrection", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "PrecomputedConstraintCorrection", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "SlidingConstraint", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "StopperConstraint", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "UncoupledConstraintCorrection", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Correction") },
    { "UniformConstraint", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "UnilateralInteractionConstraint", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.Constraint.Lagrangian.Model") },
    { "ConstraintAnimationLoop", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.AnimationLoop") },
    { "FreeMotionAnimationLoop", CreatableMoved("v22.06", "SofaConstraint", "Sofa.Component.AnimationLoop") },

    // SofaGeneralAnimationLoop was deprecated in #2635 and #2796
    { "MechanicalMatrixMapper", CreatableMoved("v22.06", "SofaGeneralAnimationLoop", "Sofa.Component.Mapping.MappedMatrix") },
    { "MultiStepAnimationLoop", CreatableMoved("v22.06", "SofaGeneralAnimationLoop", "Sofa.Component.AnimationLoop") },
    { "MultiTagAnimationLoop", CreatableMoved("v22.06", "SofaGeneralAnimationLoop", "Sofa.Component.AnimationLoop") },

    // SofaSimpleFem was deprecated in #2759
    { "HexahedronFEMForceField", CreatableMoved("v22.06", "SofaSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TetrahedronFEMForceField", CreatableMoved("v22.06", "SofaSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },

    // SofaGeneralSimpleFem was deprecated in #2759
    { "BeamFEMForceField", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedralFEMForceField", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedralFEMForceFieldAndMass", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "HexahedronFEMForceFieldAndMass", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TetrahedralCorotationalFEMForceField", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularFEMForceFieldOptim", CreatableMoved("v22.06", "SofaGeneralSimpleFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },

    // SofaMiscFem was deprecated in #2759
    { "FastTetrahedralCorotationalForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "StandardTetrahedralFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangleFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularAnisotropicFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "TriangularFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "QuadBendingFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.Elastic") },
    { "BoyceAndArruda", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "Costa", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "HyperelasticMaterial", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "MooneyRivlin", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "NeoHookean", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "Ogden", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "PlasticMaterial", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "StandardTetrahedralFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "STVenantKirchhoff", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "TetrahedronHyperelasticityFEMForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "VerondaWestman", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.HyperElastic") },
    { "TetrahedralTensorMassForceField", CreatableMoved("v22.06", "SofaMiscFem", "Sofa.Component.SolidMechanics.FEM.TensorMass") },

    // SofaDeformable was deprecated in #2759
    { "AngularSpringForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "MeshSpringForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "RestShapeSpringsForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "PolynomialRestShapeSpringsForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "SpringForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "StiffSpringForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "PolynomialSpringsForceField", CreatableMoved("v22.06", "SofaDeformable", "Sofa.Component.SolidMechanics.Spring") },

    // SofaGeneralDeformable was deprecated in #2759
    { "FastTriangularBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "FrameSpringForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "QuadBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "QuadularBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "RegularGridSpringForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangleBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangleBendingSprings", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularBiquadraticSpringsForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularQuadraticSpringsForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "VectorSpringForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.Spring") },
    { "TriangularTensorMassForceField", CreatableMoved("v22.06", "SofaGeneralDeformable", "Sofa.Component.SolidMechanics.TensorMass") },

    // SofaGeneralObjectInteraction was deprecated in #2759
    { "RepulsiveSpringForceField", CreatableMoved("v22.06", "SofaGeneralObjectInteraction", "Sofa.Component.SolidMechanics.Spring") },

    // SofaGeneralObjectInteraction was deprecated in #2790 and ...
    { "AttachConstraint", CreatableMoved("v22.06", "SofaGeneralObjectInteraction", "Sofa.Component.Constraint.Projective") },

    // SofaBoundaryCondition was deprecated in #2790 and ...
    { "AffineMovementConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedPlaneConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedRotationConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "FixedTranslationConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "HermiteSplineConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "LinearMovementConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "LinearVelocityConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "OscillatorConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ParabolicConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PartialFixedConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PartialLinearMovementConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PatchTestMovementConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PointConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "PositionBasedDynamicsConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectDirectionConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToLineConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToPlaneConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "ProjectToPointConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "AttachConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },
    { "SkeletalMotionConstraint", CreatableMoved("v22.06", "SofaBoundaryCondition", "Sofa.Component.Constraint.Projective") },

};

} // namespace sofa::helper::lifecycle
