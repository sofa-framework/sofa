#ifndef STDAFX_H
#define STDAFX_H

// To speed up compilation time for tests on Windows (using precompiled headers)
#ifdef WIN32

#include <sofa/helper/Quater.h>
#include <sofa/helper/RandomGenerator.h>
#include <sofa/defaulttype/VecTypes.h>
#include <sofa/core/ExecParams.h>

#include <SceneCreator/SceneCreator.h>
#include <SofaTest/Sofa_test.h>
#include <SofaTest/Mapping_test.h>
#include <SofaTest/Elasticity_test.h>

#include <sofa/helper/system/SetDirectory.h>
#include <sofa/helper/system/FileRepository.h>

#include <sofa/simulation/Simulation.h>
#include <SofaSimulationGraph/DAGSimulation.h>
#include <sofa/simulation/Node.h>

#include <SofaBaseMechanics/MechanicalObject.h>

#include <SofaBoundaryCondition/AffineMovementConstraint.h>
#include <SofaBoundaryCondition/QuadPressureForceField.h>
#include <SofaBoundaryCondition/TrianglePressureForceField.h>
#include <SofaBoundaryCondition/PatchTestMovementConstraint.h>

#ifdef FLEXIBLE_TEST_WITH_IMAGE
#include <image/ImageTypes.h>
#include <image/ImageContainer.h>
#endif

#include <Flexible/types/DeformationGradientTypes.h>
#include <Flexible/types/StrainTypes.h>
#include <Flexible/material/HookeForceField.h>
#include <Flexible/deformationMapping/LinearMapping.h>
#include <Flexible/strainMapping/CorotationalStrainMapping.h>
#include <Flexible/strainMapping/PrincipalStretchesMapping.h>
#include <Flexible/strainMapping/GreenStrainMapping.h>
#include <Flexible/strainMapping/InvariantMapping.h>
#include <Flexible/strainMapping/CauchyStrainMapping.h>
#include <Flexible/strainMapping/InvariantMapping.h>
#include <Flexible/strainMapping/PrincipalStretchesMapping.h>
#include <Flexible/shapeFunction/ShepardShapeFunction.h>
#include <Flexible/shapeFunction/HatShapeFunction.h>
#ifdef FLEXIBLE_TEST_WITH_IMAGE
#include <Flexible/shapeFunction/VoronoiShapeFunction.h>
#include <Flexible/shapeFunction/ShapeFunctionDiscretizer.h>
#include <Flexible/shapeFunction/DiffusionShapeFunction.h>
#endif



#endif // WIN32

#endif // STDAFX_H
