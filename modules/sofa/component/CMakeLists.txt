cmake_minimum_required(VERSION 2.8)

set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaBaseAnimationLoop" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseAnimationLoop")
RegisterDependencies("SofaBaseCollision" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseCollision")
RegisterDependencies("SofaBaseLinearSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseLinearSolver")
RegisterDependencies("SofaBaseMechanics" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseMechanics")
RegisterDependencies("SofaBaseTopology" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseTopology")
RegisterDependencies("SofaBaseVisual" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseVisual")
RegisterDependencies("SofaBoundaryCondition" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBoundaryCondition")

RegisterDependencies("SofaComponentAdvanced" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentAdvanced")
RegisterDependencies("SofaComponentCommon" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentCommon")
RegisterDependencies("SofaComponentGeneral" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentGeneral")
RegisterDependencies("SofaComponentMisc" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMisc")
RegisterDependencies("SofaComponentBase" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentBase")
RegisterDependencies("SofaComponentMain" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMain")

RegisterDependencies("SofaConstraint" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaConstraint")
RegisterDependencies("SofaDeformable" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeformable")
RegisterDependencies("SofaDenseSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDenseSolver")
RegisterDependencies("SofaEigen2Solver" OPTION EXTERNAL_HAVE_EIGEN2 PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEigen2Solver")

RegisterDependencies("SofaEngine" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEngine")
RegisterDependencies("SofaEulerianFluid" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEulerianFluid")
RegisterDependencies("SofaExplicitOdeSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExplicitOdeSolver")
RegisterDependencies("SofaExporter" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExporter")
RegisterDependencies("SofaGraphComponent" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGraphComponent")
RegisterDependencies("SofaHaptics" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaHaptics")
RegisterDependencies("SofaImplicitOdeSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaImplicitOdeSolver")
RegisterDependencies("SofaLoader" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaLoader")
RegisterDependencies("SofaMeshCollision" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMeshCollision")
RegisterDependencies("SofaMisc" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc")
RegisterDependencies("SofaMiscCollision" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscCollision")
RegisterDependencies("SofaMiscEngine" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscEngine")
RegisterDependencies("SofaMiscFem" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscFem")
RegisterDependencies("SofaMiscForceField" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscForceField")
RegisterDependencies("SofaMiscMapping" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscMapping")
RegisterDependencies("SofaMiscSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscSolver")
RegisterDependencies("SofaMiscTopology" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscTopology")
RegisterDependencies("SofaNonUniformFem" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaNonUniformFem")
RegisterDependencies("SofaObjectInteraction" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaObjectInteraction")
RegisterDependencies("SofaOpenglVisual" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaOpenglVisual") # NOT OPTION_NO_OPENGL
RegisterDependencies("SofaPardisoSolver" OPTION OPTION_PARDISO COMPILE_DEFINITIONS SOFA_HAVE_PARDISO PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPardisoSolver")
RegisterDependencies("SofaRigid" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaRigid")
RegisterDependencies("SofaSimpleFem" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimpleFem")
RegisterDependencies("SofaSparseSolver" OPTION EXTERNAL_HAVE_CSPARSE PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSparseSolver")

RegisterDependencies("SofaPreconditioner" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPreconditioner")
RegisterDependencies("SofaSphFluid" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSphFluid")
#RegisterDependencies("SofaTaucsSolver" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTaucsSolver")
RegisterDependencies("SofaTopologyMapping" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTopologyMapping")
RegisterDependencies("SofaUserInteraction" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaUserInteraction")
RegisterDependencies("SofaValidation" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaValidation")
RegisterDependencies("SofaVolumetricData" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaVolumetricData")

#include("SofaBaseAnimationLoop.cmake")
#include("SofaBaseCollision.cmake")
#include("SofaBaseLinearSolver.cmake")
#include("SofaBaseMechanics.cmake")
#include("SofaBaseTopology.cmake") 
#include("SofaBaseVisual.cmake")
#include("SofaBoundaryCondition.cmake")

#All the SofaComponent* may need something equivalent to addToBuildGroup(..) in sofa-dependencies.prf

#include("SofaComponentAdvanced.cmake") #needs: SofaVolumetricData
#include("SofaComponentCommon.cmake")
#include("SofaComponentGeneral.cmake") #needs: SofaSparseSolver SofaPreconditioner
#include("SofaComponentMisc.cmake") #needs: all SofaMisc*
#include("SofaComponentBase.cmake") #needs: SofaVolumetricData SofaNonUniformFem
#include("SofaComponentMain.cmake") #needs: SofaComponentBase

#include("SofaConstraint.cmake")
#include("SofaDeformable.cmake")
#include("SofaDenseSolver.cmake")
#if(EXTERNAL_HAVE_EIGEN2)
#    include("SofaEigen2Solver.cmake")
#endif(EXTERNAL_HAVE_EIGEN2)
#include("SofaEngine.cmake")
#include("SofaEulerianFluid.cmake")
#include("SofaExplicitOdeSolver.cmake")
#include("SofaExporter.cmake") 
#include("SofaGraphComponent.cmake")
#include("SofaHaptics.cmake") 
#include("SofaImplicitOdeSolver.cmake")
#include("SofaLoader.cmake")
#include("SofaMeshCollision.cmake")
#include("SofaMisc.cmake") 
#include("SofaMiscCollision.cmake") #needs: SofaConstriant SofaVolumetricData
#include("SofaMiscEngine.cmake") #needs: SofaNonUniformFem
#include("SofaMiscFem.cmake") 
#include("SofaMiscForceField.cmake") 
#include("SofaMiscMapping.cmake") #missing automatic compiler optimization flags
#include("SofaMiscSolver.cmake")
#include("SofaMiscTopology.cmake")
#include("SofaNonUniformFem.cmake") #needs: SofaVolumetricData 
#include("SofaObjectInteraction.cmake")
#if(NOT OPTION_NO_OPENGL)
#  include("SofaOpenglVisual.cmake")
#endif()
#if(OPTION_PARDISO)
#    include("SofaPardisoSolver.cmake")#endif(OPTION_PARDISO)
#include("SofaRigid.cmake")
#include("SofaSimpleFem.cmake") 
#if(EXTERNAL_HAVE_CSPARSE)
#    include("SofaSparseSolver.cmake")
#endif(EXTERNAL_HAVE_CSPARSE)
#include("SofaPreconditioner.cmake")
#include("SofaSphFluid.cmake")
##include("SofaTaucsSolver.cmake") #missing LIBS *= -lblas_win32_MT, etc..
#include("SofaTopologyMapping.cmake")
#include("SofaUserInteraction.cmake")
#include("SofaValidation.cmake") 
#include("SofaVolumetricData.cmake")

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
