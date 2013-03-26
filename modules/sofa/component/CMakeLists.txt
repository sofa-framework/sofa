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
include("SofaUserInteraction.cmake")
#include("SofaConstraint.cmake") #having eigen problems?
#include("SofaHaptics.cmake") #needs: SofaConstraint
#include("SofaSparseSolver.cmake") #needs: metis 
#include("SofaPreconditioner.cmake") #needs: SofaSparseSolver 
#include("SofaVolumetricData.cmake") #needs: SofaFlowVR
#include("SofaNonUniformFem.cmake") #needs: SofaVolumetricData 
include("SofaMiscForcefield.cmake") 
include("SofaMiscFem.cmake") 
#include("SofaMiscEngine.cmake") #needs: SofaNonUniformFem
#include("SofaMiscCollision.cmake") #needs: SofaConstriant SofaVolumetricData
include("SofaMisc.cmake") 
include("SofaValidation.cmake") 
include("SofaExporter.cmake") 
include("SofaLoader.cmake")
include("SofaMiscMapping.cmake") #missing automatic compiler optimization flags
#include("SofaTaucsSolver.cmake") #missing LIBS *= -lblas_win32_MT, etc..
include("SofaTopologyMapping.cmake")
include("SofaOpenglVisual.cmake")
include("SofaSimpleFem.cmake") 

#All the SofaComponent* may need something equivalent to addToBuildGroup(..) in sofa-dependencies.prf

#include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
include("SofaComponentCommon.cmake")
#include("SofaComponent.cmake") #needs: SofaComponentBase
#include("SofaComponentGeneral.cmake") #needs: SofaConstraint SofaHaptics SofaSparseSolver SofaPreconditioner
#include("SofaComponentMisc.cmake") #needs: all SofaMisc*
#include("SofaComponentBase.cmake") #needs: SofaVolumetricData SofaNonUniformFem


include(${SOFA_CMAKE_DIR}/post.cmake)
