cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("SofaBaseAnimationLoop.cmake")
include("SofaBaseLinearSolver.cmake")
include("SofaBaseTopology.cmake") 
include("SofaBaseMechanics.cmake")
include("SofaBaseCollision.cmake")
include("SofaBaseVisual.cmake")
include("SofaSphFluid.cmake")
include("SofaDeformable.cmake")
include("SofaExplicitOdeSolver.cmake")
include("SofaImplicitOdeSolver.cmake")
include("SofaGraphComponent.cmake")
include("SofaEulerianFluid.cmake")
include("SofaMiscSolver.cmake")
include("SofaObjectInteraction.cmake")
include("SofaMeshCollision.cmake")
include("SofaEngine.cmake")
include("SofaPardisoSolver.cmake") 
include("SofaRigid.cmake")
include("SofaEigen2Solver.cmake")         #needs: eigen-unsupported ? (what is that?)
include("SofaBoundaryCondition.cmake")
include("SofaDenseSolver.cmake")
include("SofaMiscTopology.cmake")
#include("SofaUserInteraction.cmake") #needs: SofaTopologyMapping
#include("SofaConstraint.cmake") #needs: SofaSimpleFem SofaUserInteraction
#include("SofaHaptics.cmake") #needs: SofaConstraint
#include("SofaSparseSolver.cmake") #needs: metis 
#include("SofaPreconditioner.cmake") #needs: SofaSparseSolver SofaSimpleFem
#include("SofaVolumetricData.cmake") #needs: SofaUserInteraction 
#include("SofaNonUniformFem.cmake") #needs: SofaVolumetricData SofaOpenglVisual SofaSimpleFem
include("SofaMiscForcefield.cmake") 
#include("SofaMiscFem.cmake") #needs: SofaSimpleFem
#include("SofaMiscEngine.cmake") #needs: SofaNonUniformFem
#include("SofaMiscCollision.cmake") #needs: SofaConstriant SofaVolumetricData
#include("SofaMisc.cmake") #needs: SofaTopologyMapping
include("SofaValidation.cmake") 
include("SofaExporter.cmake") 
include("SofaLoader.cmake")
#include("SofaMiscMapping.cmake") #needs: SofaSimpleFem SofaOpenglVisual SofaUserInteraction #missing automatic compiler optimization flags
#include("SofaTaucsSolver.cmake") #missing LIBS *= -lblas_win32_MT, etc..

#include("SofaOpenglVisual.cmake") #link needs: SofaSimulation
#include("SofaSimpleFem.cmake") #needs: SofaOpenglVisual
include("SofaTopologyMapping.cmake")
#include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
#include("SofaComponentCommon.cmake") #needs: sofa_loader sofa_simple_fem
#include("SofaComponent.cmake") #needs: sofa_component_base
#include("SofaComponentGeneral.cmake") #needs: a lot of stuff
#include("SofaComponentMisc.cmake") #needs: all SofaMisc*


include(${SOFA_CMAKE_DIR}/post.cmake)
