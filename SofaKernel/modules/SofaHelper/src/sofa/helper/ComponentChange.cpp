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


namespace sofa
{
namespace helper
{
namespace lifecycle
{

std::map<std::string, Deprecated> deprecatedComponents = {
    // SofaMiscForceField
    {"LennardJonesForceField", Deprecated("v17.12", "v18.12")},
    {"MatrixMass", Deprecated("v19.06", "v19.12")},

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
    { "LennardJonesForceField", Pluginized("v20.12", "SofaMiscForceField") },

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
    { "LengthContainer", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "PoissonContainer", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "RadiusContainer", Pluginized("v20.12", "SofaGeneralSimpleFem") },
    { "StiffnessContainer", Pluginized("v20.12", "SofaGeneralSimpleFem") },
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
    { "DistanceLMConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "DistanceLMContactConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "DOFBlockerLMConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "FixedLMConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "FreeMotionAnimationLoop", Pluginized("v20.12", "SofaConstraint") },
    { "GenericConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "GenericConstraintSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LCPConstraintSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LinearSolverConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "LMConstraintDirectSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LMConstraintSolver", Pluginized("v20.12", "SofaConstraint") },
    { "LMDNewProximityIntersection", Pluginized("v20.12", "SofaConstraint") },
    { "LocalMinDistance", Pluginized("v20.12", "SofaConstraint") },
    { "MappingGeometricStiffnessForceField", Pluginized("v20.12", "SofaConstraint") },
    { "PrecomputedConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "SlidingConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "StopperConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "UncoupledConstraintCorrection", Pluginized("v20.12", "SofaConstraint") },
    { "UniformConstraint", Pluginized("v20.12", "SofaConstraint") },
    { "UnilateralInteractionConstraint", Pluginized("v20.12", "SofaConstraint") },

    /***********************/
    // REMOVED SINCE v20.06

    // SofaKernel
    {"Point", Removed("v19.12", "v20.06")},
    {"TPointModel", Removed("v19.12", "v20.06")},
    {"PointModel", Removed("v19.12", "v20.06")},
    {"PointMesh", Removed("v19.12", "v20.06")},
    {"PointSet", Removed("v19.12", "v20.06")},

    {"Line", Removed("v19.12", "v20.06")},
    {"TLineModel", Removed("v19.12", "v20.06")},
    {"LineMeshModel", Removed("v19.12", "v20.06")},
    {"LineSetModel", Removed("v19.12", "v20.06")},
    {"LineMesh", Removed("v19.12", "v20.06")},
    {"LineSet", Removed("v19.12", "v20.06")},
    {"LineModel", Removed("v19.12", "v20.06")},

    {"Triangle", Removed("v19.12", "v20.06")},
    {"TriangleSet", Removed("v19.12", "v20.06")},
    {"TriangleMesh", Removed("v19.12", "v20.06")},
    {"TriangleSetModel", Removed("v19.12", "v20.06")},
    {"TriangleMeshModel", Removed("v19.12", "v20.06")},
    {"TriangleModel", Removed("v19.12", "v20.06")},
    {"TTriangleModel", Removed("v19.12", "v20.06")},

    {"Sphere", Removed("v19.12", "v20.06")},
    {"SphereModel", Removed("v19.12", "v20.06")},
    {"TSphereModel", Removed("v19.12", "v20.06")},

    {"Capsule", Removed("v19.12", "v20.06")},
    {"CapsuleModel", Removed("v19.12", "v20.06")},
    {"TCapsuleModel", Removed("v19.12", "v20.06")},

    {"RigidCapsule", Removed("v19.12", "v20.06")},
    {"CapsuleModel", Removed("v19.12", "v20.06")},

    {"Cube", Removed("v19.12", "v20.06")},
    {"CubeModel", Removed("v19.12", "v20.06")},

    {"CudaPoint", Removed("v19.12", "v20.06")},
    {"CudaPointModel", Removed("v19.12", "v20.06")},

    {"Cylinder", Removed("v19.12", "v20.06")},
    {"CylinderModel", Removed("v19.12", "v20.06")},

    {"Ray", Removed("v19.12", "v20.06")},
    {"RayModel", Removed("v19.12", "v20.06")},

    {"Tetrahedron", Removed("v19.12", "v20.06")},
    {"TetrahedronModel", Removed("v19.12", "v20.06")},

    {"Euler", Removed("v19.12", "v20.06")},
    {"EulerExplicit", Removed("v19.12", "v20.06")},
    {"ExplicitEuler", Removed("v19.12", "v20.06")},
    {"EulerSolver", Removed("v19.12", "v20.06")},
    {"ExplicitEulerSolver", Removed("v19.12", "v20.06")},

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

