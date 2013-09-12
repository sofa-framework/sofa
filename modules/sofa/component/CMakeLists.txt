set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

include(${SOFA_CMAKE_DIR}/preSolution.cmake)

RegisterDependencies("SofaBaseAnimationLoop" OPTION "SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseAnimationLoop")
RegisterDependencies("SofaBaseCollision" OPTION "SOFA-LIB_COMPONENT_BASE_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseCollision")
RegisterDependencies("SofaBaseLinearSolver" OPTION "SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseLinearSolver")
RegisterDependencies("SofaBaseMechanics" OPTION "SOFA-LIB_COMPONENT_BASE_MECHANICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseMechanics")
RegisterDependencies("SofaBaseTopology" OPTION "SOFA-LIB_COMPONENT_BASE_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseTopology")
RegisterDependencies("SofaBaseVisual" OPTION "SOFA-LIB_COMPONENT_BASE_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseVisual")

RegisterDependencies("SofaComponentAdvanced" OPTION "SOFA-LIB_COMPONENT_COMPONENT_ADVANCED" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentAdvanced")
RegisterDependencies("SofaComponentCommon" OPTION "SOFA-LIB_COMPONENT_COMPONENT_COMMON" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentCommon")
RegisterDependencies("SofaComponentGeneral" OPTION "SOFA-LIB_COMPONENT_COMPONENT_GENERAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentGeneral")
RegisterDependencies("SofaComponentMisc" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMisc")
RegisterDependencies("SofaComponentBase" OPTION "SOFA-LIB_COMPONENT_COMPONENT_BASE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentBase")
RegisterDependencies("SofaComponentMain" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MAIN" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMain")

RegisterDependencies("SofaBoundaryCondition" OPTION "SOFA-LIB_COMPONENT_BOUNDARY_CONDITION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBoundaryCondition")
RegisterDependencies("SofaConstraint" OPTION "SOFA-LIB_COMPONENT_CONSTRAINT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaConstraint")
RegisterDependencies("SofaDeformable" OPTION "SOFA-LIB_COMPONENT_DEFORMABLE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeformable")
RegisterDependencies("SofaDenseSolver" OPTION "SOFA-LIB_COMPONENT_DENSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDenseSolver")
RegisterDependencies("SofaEigen2Solver" OPTION "SOFA-LIB_COMPONENT_EIGEN2_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEigen2Solver")

RegisterDependencies("SofaEngine" OPTION "SOFA-LIB_COMPONENT_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEngine")
RegisterDependencies("SofaEulerianFluid" OPTION "SOFA-LIB_COMPONENT_EULERIAN_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEulerianFluid")
RegisterDependencies("SofaExplicitOdeSolver" OPTION "SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExplicitOdeSolver")
RegisterDependencies("SofaExporter" OPTION "SOFA-LIB_COMPONENT_EXPORTER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExporter")
RegisterDependencies("SofaGraphComponent" OPTION "SOFA-LIB_COMPONENT_GRAPH_COMPONENT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGraphComponent")
RegisterDependencies("SofaHaptics" OPTION "SOFA-LIB_COMPONENT_HAPTICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaHaptics")
RegisterDependencies("SofaImplicitOdeSolver" OPTION "SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaImplicitOdeSolver")
RegisterDependencies("SofaLoader" OPTION "SOFA-LIB_COMPONENT_LOADER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaLoader")
RegisterDependencies("SofaMeshCollision" OPTION "SOFA-LIB_COMPONENT_MESH_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMeshCollision")
RegisterDependencies("SofaMisc" OPTION "SOFA-LIB_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc")
RegisterDependencies("SofaMiscCollision" OPTION "SOFA-LIB_COMPONENT_MISC_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscCollision")
RegisterDependencies("SofaMiscEngine" OPTION "SOFA-LIB_COMPONENT_MISC_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscEngine")
RegisterDependencies("SofaMiscFem" OPTION "SOFA-LIB_COMPONENT_MISC_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscFem")
RegisterDependencies("SofaMiscForceField" OPTION "SOFA-LIB_COMPONENT_MISC_FORCEFIELD" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscForceField")
RegisterDependencies("SofaMiscMapping" OPTION "SOFA-LIB_COMPONENT_MISC_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscMapping")
RegisterDependencies("SofaMiscSolver" OPTION "SOFA-LIB_COMPONENT_MISC_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscSolver")
RegisterDependencies("SofaMiscTopology" OPTION "SOFA-LIB_COMPONENT_MISC_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscTopology")
RegisterDependencies("SofaNonUniformFem" OPTION "SOFA-LIB_COMPONENT_NON_UNIFORM_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaNonUniformFem")
RegisterDependencies("SofaObjectInteraction" OPTION "SOFA-LIB_COMPONENT_OBJECT_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaObjectInteraction")
RegisterDependencies("SofaOpenglVisual" OPTION "SOFA-LIB_COMPONENT_OPENGL_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaOpenglVisual") # NOT SOFA-MISC_NO_OPENGL

RegisterDependencies("SofaPardisoSolver" OPTION "SOFA-LIB_COMPONENT_PARDISO_SOLVER" COMPILE_DEFINITIONS SOFA_HAVE_PARDISO PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPardisoSolver") # previous option was : SOFA-LIB_PARDISO_SOLVER

RegisterDependencies("SofaRigid" OPTION "SOFA-LIB_COMPONENT_RIGID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaRigid")
RegisterDependencies("SofaSimpleFem" OPTION "SOFA-LIB_COMPONENT_SIMPLE_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimpleFem")

RegisterDependencies("SofaSparseSolver" OPTION "SOFA-LIB_COMPONENT_SPARSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSparseSolver")

RegisterDependencies("SofaPreconditioner" OPTION "SOFA-LIB_COMPONENT_PRECONDITIONER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPreconditioner")
RegisterDependencies("SofaSphFluid" OPTION "SOFA-LIB_COMPONENT_SPH_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSphFluid")
#RegisterDependencies("SofaTaucsSolver" OPTION "SOFA-LIB_COMPONENT_TAUCS_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTaucsSolver") #missing LIBS *= -lblas_win32_MT, etc..
RegisterDependencies("SofaTopologyMapping" OPTION "SOFA-LIB_COMPONENT_TOPOLOGY_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTopologyMapping")
RegisterDependencies("SofaUserInteraction" OPTION "SOFA-LIB_COMPONENT_USER_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaUserInteraction")
RegisterDependencies("SofaValidation" OPTION "SOFA-LIB_COMPONENT_VALIDATION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaValidation")
RegisterDependencies("SofaVolumetricData" OPTION "SOFA-LIB_COMPONENT_VOLUMETRIC_DATA" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaVolumetricData")

include(${SOFA_CMAKE_DIR}/postSolution.cmake)
