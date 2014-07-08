#ifndef STDAFX_H
#define STDAFX_H

// To speed up compilation time for tests on Windows (using precompiled headers)
#ifdef WIN32

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>
#include <sofa/component/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including constraint, force and mass
#include <sofa/component/projectiveconstraintset/AffineMovementConstraint.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/component/interactionforcefield/MeshSpringForceField.h>
#include <sofa/component/forcefield/TetrahedronFEMForceField.h>
#include <sofa/core/MechanicalParams.h>

#include <sofa/defaulttype/VecTypes.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include "Sofa_test.h"
#include <sofa/component/collision/BarycentricContactMapper.h>
#include <sofa/component/mapping/BarycentricMapping.h>

#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/component/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including constraint, force and mass
#include <sofa/component/topology/BezierTetrahedronSetTopologyContainer.h>
#include <sofa/component/topology/BezierTetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/component/mass/MeshMatrixMass.h>
#include <sofa/component/engine/GenerateCylinder.h>
#include <sofa/component/topology/Mesh2BezierTopologicalMapping.h>
#include <sofa/defaulttype/VecTypes.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/simulation/common/Visitor.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/tree/TreeSimulation.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/simulation/common/Simulation.h>

#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <sofa/component/forcefield/TetrahedralTensorMassForceField.h>
#include <sofa/component/forcefield/TetrahedralCorotationalFEMForceField.h>
#include <sofa/component/topology/TopologySparseData.inl>
#include <sofa/component/forcefield/TrianglePressureForceField.h>
#include <sofa/component/projectiveconstraintset/AffineMovementConstraint.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/engine/PairBoxRoi.h>
#include <sofa/component/odesolver/StaticSolver.h>
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.h>

#include <sofa/component/init.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/common/SceneLoaderXML.h>

#include <gtest/gtest.h>
#include "Sofa_test.h"

#include <sofa/component/linearsolver/EigenSparseMatrix.h>
#include <sofa/component/linearsolver/SparseMatrix.h>
#include <sofa/component/linearsolver/CompressedRowSparseMatrix.h>
#include <sofa/component/linearsolver/FullMatrix.h>
#include <sofa/component/linearsolver/FullVector.h>
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

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/mapping/DistanceMapping.h>
#include <sofa/component/mapping/DistanceFromTargetMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include "../../../applications/tutorials/objectCreator/ObjectCreator.h"


#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/collision/MeshIntTool.h>

#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/component/topology/PointSetTopologyContainer.h>
#include <sofa/component/projectiveconstraintset/ProjectToPlaneConstraint.h>
#include <sofa/component/container/MechanicalObject.h>
#include <sofa/core/MechanicalParams.h>
#include <sofa/defaulttype/VecTypes.h>

#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/component/init.h>
#include <plugins/SceneCreator/SceneCreator.h>

#include <sofa/component/container/MechanicalObject.h>

//Force field
#include <sofa/component/forcefield/QuadPressureForceField.h>

// topology
#include <sofa/component/topology/RegularGridTopology.h>

#include <sofa/component/engine/BoxROI.h>

// Constraint
#include <sofa/component/projectiveconstraintset/ProjectToLineConstraint.h>
#include <sofa/component/projectiveconstraintset/FixedConstraint.h>
#include <sofa/component/projectiveconstraintset/FixedPlaneConstraint.h>

//Solver
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/odesolver/StaticSolver.h>

#include "Mapping_test.h"
#include <sofa/component/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/component/mapping/RigidMapping.h>
#include <sofa/component/container/MechanicalObject.h>

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

#include <sofa/component/init.h>
#include <sofa/component/mapping/SubsetMultiMapping.h>
#include <sofa/component/mapping/DistanceMapping.h>
#include <sofa/component/mapping/DistanceFromTargetMapping.h>
#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/topology/EdgeSetTopologyContainer.h>
#include <sofa/component/collision/SphereModel.h>
#include <sofa/component/topology/CubeTopology.h>
#include <sofa/component/visualmodel/VisualStyle.h>
#include <sofa/component/odesolver/EulerImplicitSolver.h>
#include <sofa/component/odesolver/EulerSolver.h>
#include <sofa/component/linearsolver/CGLinearSolver.h>
#include <sofa/component/collision/OBBModel.h>
#include <sofa/simulation/tree/tree.h>
#include <sofa/simulation/tree/TreeSimulation.h>

//Using double by default, if you have SOFA_FLOAT in use in you sofa-default.cfg, then it will be FLOAT.
#include <sofa/component/typedef/Sofa_typedef.h>
//#include <plugins/SceneCreator/SceneCreator.h>


#include <sofa/simulation/common/Simulation.h>
#include <sofa/component/collision/DefaultCollisionGroupManager.h>
#include <sofa/simulation/tree/GNode.h>

#include <sofa/component/topology/MeshTopology.h>
#include <sofa/component/collision/MeshIntTool.h>
#include <sofa/component/collision/MeshMinProximityIntersection.h>
#include <sofa/component/collision/MeshNewProximityIntersection.inl>

#include <sofa/component/init.h>
//#include <plugins/SceneCreator/SceneCreator.h>
//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>
#include <sofa/helper/set.h>
// Including constraint, force and mass
#include <sofa/component/topology/TetrahedronSetGeometryAlgorithms.h>
#include <sofa/component/topology/CommonAlgorithms.h>
#include <sofa/defaulttype/VecTypes.h>
#include <ctime>
#include <plugins/SceneCreator/SceneCreator.h>

#endif // WIN32

#endif // STDAFX_H