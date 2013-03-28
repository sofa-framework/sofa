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

include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
include("SofaComponentCommon.cmake")
include("SofaComponentGeneral.cmake") #needs: SofaSparseSolver SofaPreconditioner
include("SofaComponentMisc.cmake") #needs: all SofaMisc*
include("SofaComponentBase.cmake") #needs: SofaVolumetricData SofaNonUniformFem
include("SofaComponentMain.cmake") #needs: SofaComponentBase

include("SofaConstraint.cmake")
include("SofaDeformable.cmake")
include("SofaDenseSolver.cmake")
if(EXTERNAL_HAVE_EIGEN2)
    include("SofaEigen2Solver.cmake")
endif(EXTERNAL_HAVE_EIGEN2)
include("SofaEngine.cmake")
include("SofaEulerianFluid.cmake")
include("SofaExplicitOdeSolver.cmake")
include("SofaExporter.cmake") 
include("SofaGraphComponent.cmake")
include("SofaHaptics.cmake") 
include("SofaImplicitOdeSolver.cmake")
include("SofaLoader.cmake")
include("SofaMeshCollision.cmake")
include("SofaMisc.cmake") 
include("SofaMiscCollision.cmake") #needs: SofaConstriant SofaVolumetricData
include("SofaMiscEngine.cmake") #needs: SofaNonUniformFem
include("SofaMiscFem.cmake") 
include("SofaMiscForceField.cmake") 
include("SofaMiscMapping.cmake") #missing automatic compiler optimization flags
include("SofaMiscSolver.cmake")
include("SofaMiscTopology.cmake")
include("SofaNonUniformFem.cmake") #needs: SofaVolumetricData 
include("SofaObjectInteraction.cmake")
include("SofaOpenglVisual.cmake")
if(OPTION_PARDISO)
    include("SofaPardisoSolver.cmake")
endif(OPTION_PARDISO)
include("SofaRigid.cmake")
include("SofaSimpleFem.cmake") 
if(EXTERNAL_HAVE_CSPARSE)
    include("SofaSparseSolver.cmake") #needs: metis
    include("SofaPreconditioner.cmake")
endif(EXTERNAL_HAVE_CSPARSE)
include("SofaSphFluid.cmake")
#include("SofaTaucsSolver.cmake") #missing LIBS *= -lblas_win32_MT, etc..
include("SofaTopologyMapping.cmake")
include("SofaUserInteraction.cmake")
include("SofaValidation.cmake") 
include("SofaVolumetricData.cmake")

include(${SOFA_CMAKE_DIR}/post.cmake)
