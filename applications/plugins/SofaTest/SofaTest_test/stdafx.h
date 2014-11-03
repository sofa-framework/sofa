#ifndef STDAFX_H
#define STDAFX_H

// To speed up compilation time for tests on Windows (using precompiled headers)
#ifdef WIN32

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including constraint, force and mass
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <SofaDeformable/MeshSpringForceField.h>
#include <SofaSimpleFem/TetrahedronFEMForceField.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include "Sofa_test.h"
#include <SofaMeshCollision/BarycentricContactMapper.h>
#include <SofaBaseMechanics/BarycentricMapping.h>

#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including constraint, force and mass
#include <SofaBaseTopology/BezierTetrahedronSetTopologyContainer.h>
#include <SofaBaseTopology/BezierTetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <plugins/SceneCreator/SceneCreator.h>
#include <SofaMiscForceField/MeshMatrixMass.h>
#include <SofaEngine/GenerateCylinder.h>
#include <SofaTopologyMapping/Mesh2BezierTopologicalMapping.h>
#include <sofa/defaulttype/VecTypes.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/tree/TreeSimulation.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <sofa/simulation/common/Simulation.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <SofaMiscFem/TetrahedralTensorMassForceField.h>
#include <SofaSimpleFem/TetrahedralCorotationalFEMForceField.h>
#include <SofaBaseTopology/TopologySparseData.inl>
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaEngine/PairBoxRoi.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>

#include <SofaComponentMain/init.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

#include <gtest/gtest.h>
#include "Sofa_test.h"

#include <SofaEigen2Solver/EigenSparseMatrix.h>
#include <SofaBaseLinearSolver/SparseMatrix.h>
#include <SofaBaseLinearSolver/CompressedRowSparseMatrix.h>
#include <SofaBaseLinearSolver/FullMatrix.h>
#include <SofaBaseLinearSolver/FullVector.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Vec.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/helper/RandomGenerator.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include <sofa/simulation/common/Node.h>

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentMain/init.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaMiscMapping/DistanceMapping.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseCollision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"


#include <sofa/simulation/common/Simulation.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/MeshIntTool.h>

#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <SofaBaseTopology/PointSetTopologyContainer.h>
#include <SofaBoundaryCondition/ProjectToPlaneConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <SofaComponentMain/init.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include <SofaBaseMechanics/MechanicalObject.h>

//Force field
#include <SofaBoundaryCondition/QuadPressureForceField.h>
// Base class
#include "ForceField_test.h"
//Force field
#include <SofaBoundaryCondition/TrianglePressureForceField.h>

// topology
#include <SofaBaseTopology/RegularGridTopology.h>

#include <SofaEngine/BoxROI.h>

// Constraint
#include <SofaBoundaryCondition/ProjectToLineConstraint.h>
#include <SofaBoundaryCondition/FixedConstraint.h>
#include <SofaBoundaryCondition/FixedPlaneConstraint.h>

//Solver
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaImplicitOdeSolver/StaticSolver.h>

#include "Mapping_test.h"
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <SofaRigid/RigidMapping.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sofa/helper/ArgumentParser.h>
#include <sofa/helper/UnitTest.h>
#include <sofa/helper/vector_algebra.h>
#include <sofa/helper/vector.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/helper/system/PluginManager.h>

//#include <sofa/simulation/tree/TreeSimulation.h>
#ifdef SOFA_HAVE_DAG
#include <sofa/simulation/graph/DAGSimulation.h>
#endif
#include <sofa/simulation/common/Node.h>
#include <sofa/simulation/common/xml/initXml.h>

#include <sofa/gui/GUIManager.h>
#include <sofa/gui/Main.h>
#include <sofa/helper/system/FileRepository.h>

#include <SofaComponentMain/init.h>
#include <SofaMiscMapping/SubsetMultiMapping.h>
#include <SofaMiscMapping/DistanceMapping.h>
#include <SofaMiscMapping/DistanceFromTargetMapping.h>
#include <SofaBaseTopology/MeshTopology.h>
#include <SofaBaseTopology/EdgeSetTopologyContainer.h>
#include <SofaBaseCollision/SphereModel.h>
#include <SofaBaseTopology/CubeTopology.h>
#include <SofaBaseVisual/VisualStyle.h>
#include <SofaImplicitOdeSolver/EulerImplicitSolver.h>
#include <SofaExplicitOdeSolver/EulerSolver.h>
#include <SofaBaseLinearSolver/CGLinearSolver.h>
#include <SofaBaseCollision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>


#include <sofa/simulation/common/Simulation.h>
#include <SofaMiscCollision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <SofaBaseTopology/MeshTopology.h>
#include <SofaMeshCollision/MeshIntTool.h>
#include <SofaMeshCollision/MeshMinProximityIntersection.h>
#include <SofaMeshCollision/MeshNewProximityIntersection.inl>

#include <SofaComponentMain/init.h>
//#include <plugins/SceneCreator/SceneCreator.h>
//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/set.h>
// Including constraint, force and mass
#include <SofaBaseTopology/TetrahedronSetGeometryAlgorithms.h>
#include <SofaBaseTopology/CommonAlgorithms.h>
#include <sofa/defaulttype/VecTypes.h>
#include <ctime>
#include <plugins/SceneCreator/SceneCreator.h>

#include "Sofa_test.h"
#include <SofaEngine/TestEngine.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include <sofa/core/objectmodel/Data.h>

#endif // WIN32

#endif // STDAFX_H
