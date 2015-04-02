#ifndef STDAFX_H
#define STDAFX_H

// To speed up compilation time for tests on Windows (using precompiled headers)
#ifdef WIN32

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

// Including component
#include "../deformationMapping/LinearMapping.h"

#include <Mapping_test.h>

#include <plugins/SofaTest/Sofa_test.h>
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including component
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <plugins/SofaTest/Sofa_test.h>
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including component
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include "Elasticity_test.h"
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <SofaBoundaryCondition/QuadPressureForceField.h>
#include "../material/HookeForceField.h"
#include <SofaBaseMechanics/MechanicalObject.h>

#include "Elasticity_test.h"
#include <plugins/SceneCreator/SceneCreator.h>
#include <sofa/defaulttype/VecTypes.h>

//Including Simulation
#include <SofaComponentMain/init.h>
#include <sofa/simulation/graph/DAGSimulation.h>

#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include "../material/HookeForceField.h"
#include <SofaBaseMechanics/MechanicalObject.h>


#include <plugins/SofaTest/Sofa_test.h>
#include<sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>
#include <SofaComponentMain/init.h>
#include <sofa/core/ExecParams.h>

//Including Simulation
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/common/Node.h>

// Including component
#include <SofaBoundaryCondition/PatchTestMovementConstraint.h>
#include <SofaBaseMechanics/MechanicalObject.h>

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

// Including component

#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include "../deformationMapping/LinearMapping.h"

#include <Mapping_test.h>

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>

// Including component
#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include "../deformationMapping/LinearMapping.h"

#include <Mapping_test.h>

#include <sofa/helper/Quater.h>

#include "../types/DeformationGradientTypes.h"
#include "../types/StrainTypes.h"

#include "../strainMapping/CorotationalStrainMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"
#include "../strainMapping/GreenStrainMapping.h"
#include "../strainMapping/InvariantMapping.h"
#include "../strainMapping/CauchyStrainMapping.h"
#include "../strainMapping/InvariantMapping.h"
#include "../strainMapping/PrincipalStretchesMapping.h"

#include <image/ImageTypes.h>
#include <image/ImageContainer.h>
#include "../shapeFunction/VoronoiShapeFunction.h"
#include "../shapeFunction/ShepardShapeFunction.h"
#include "../shapeFunction/HatShapeFunction.h"
#include "../shapeFunction/ShapeFunctionDiscretizer.h"
#include "../shapeFunction/DiffusionShapeFunction.h"

#include <Mapping_test.h>


#endif // WIN32

#endif // STDAFX_H
