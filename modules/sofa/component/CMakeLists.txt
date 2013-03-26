cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/pre.cmake)

include("SofaBaseAnimationLoop.cmake")
include("SofaBaseCollision.cmake")
include("SofaBaseLinearSolver.cmake")
include("SofaBaseMechanics.cmake")
include("SofaBaseTopology.cmake") 
include("SofaBaseVisual.cmake")
include("SofaBoundaryCondition.cmake")

#All the SofaComponent* may need something equivalent to addToBuildGroup(..) in sofa-dependencies.prf

#include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
include("SofaComponentCommon.cmake")
#include("SofaComponent.cmake") #needs: SofaComponentBase
#include("SofaComponentGeneral.cmake") #needs: SofaConstraint SofaHaptics SofaSparseSolver SofaPreconditioner
#include("SofaComponentMisc.cmake") #needs: all SofaMisc*
#include("SofaComponentBase.cmake") #needs: SofaVolumetricData SofaNonUniformFem

#include("SofaConstraint.cmake") #having eigen problems?
include("SofaDeformable.cmake")
include("SofaDenseSolver.cmake")
include("SofaEigen2Solver.cmake")         #needs: eigen-unsupported ? (what is that?)
include("SofaEngine.cmake")
include("SofaEulerianFluid.cmake")
include("SofaExplicitOdeSolver.cmake")
include("SofaExporter.cmake") 
include("SofaGraphComponent.cmake")
#include("SofaHaptics.cmake") #needs: SofaConstraint
include("SofaImplicitOdeSolver.cmake")
include("SofaLoader.cmake")
include("SofaMeshCollision.cmake")
include("SofaMisc.cmake") 
#include("SofaMiscCollision.cmake") #needs: SofaConstriant SofaVolumetricData
#include("SofaMiscEngine.cmake") #needs: SofaNonUniformFem
include("SofaMiscFem.cmake") 
include("SofaMiscForcefield.cmake") 
include("SofaMiscMapping.cmake") #missing automatic compiler optimization flags
include("SofaMiscSolver.cmake")
include("SofaMiscTopology.cmake")
#include("SofaNonUniformFem.cmake") #needs: SofaVolumetricData 
include("SofaObjectInteraction.cmake")
include("SofaOpenglVisual.cmake")
include("SofaPardisoSolver.cmake") 
#include("SofaPreconditioner.cmake") #needs: SofaSparseSolver 
include("SofaRigid.cmake")
include("SofaSimpleFem.cmake") 
#include("SofaSparseSolver.cmake") #needs: metis 
include("SofaSphFluid.cmake")
#include("SofaTaucsSolver.cmake") #missing LIBS *= -lblas_win32_MT, etc..
include("SofaTopologyMapping.cmake")
include("SofaUserInteraction.cmake")
include("SofaValidation.cmake") 
#include("SofaVolumetricData.cmake") #needs: SofaFlowVR

include(${SOFA_CMAKE_DIR}/post.cmake)
