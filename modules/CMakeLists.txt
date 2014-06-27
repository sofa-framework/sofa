set(SOLUTION_NAME "SofaComponent")
project(${SOLUTION_NAME})

RegisterProjects("SofaBaseAnimationLoop" OPTION "SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseAnimationLoop")
#RegisterProjects("SofaBaseAnimationLoop_test" OPTION "SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseAnimationLoop/SofaBaseAnimationLoop_test")

RegisterProjects("SofaBaseCollision" OPTION "SOFA-LIB_COMPONENT_BASE_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseCollision")
#RegisterProjects("SofaBaseCollision_test" OPTION "SOFA-LIB_COMPONENT_BASE_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseCollision/SofaBaseCollision_test")

RegisterProjects("SofaBaseLinearSolver" OPTION "SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseLinearSolver")
#RegisterProjects("SofaBaseLinearSolver_test" OPTION "SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseLinearSolver/SofaBaseLinearSolver_test")

RegisterProjects("SofaBaseMechanics" OPTION "SOFA-LIB_COMPONENT_BASE_MECHANICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseMechanics")
RegisterProjects("SofaBaseMechanics_test" OPTION "SOFA-LIB_COMPONENT_BASE_MECHANICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseMechanics/SofaBaseMechanics_test")

RegisterProjects("SofaBaseTopology" OPTION "SOFA-LIB_COMPONENT_BASE_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseTopology")
#RegisterProjects("SofaBaseTopology_test" OPTION "SOFA-LIB_COMPONENT_BASE_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseTopology/SofaBaseTopology_test")

RegisterProjects("SofaBaseVisual" OPTION "SOFA-LIB_COMPONENT_BASE_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseVisual")
RegisterProjects("SofaBaseVisual_test" OPTION "SOFA-LIB_COMPONENT_BASE_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBaseVisual/SofaBaseVisual_test")

RegisterProjects("SofaComponentAdvanced" OPTION "SOFA-LIB_COMPONENT_COMPONENT_ADVANCED" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentAdvanced")
#RegisterProjects("SofaComponentAdvanced_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_ADVANCED" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentAdvanced/SofaComponentAdvanced_test")

RegisterProjects("SofaComponentCommon" OPTION "SOFA-LIB_COMPONENT_COMPONENT_COMMON" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentCommon")
#RegisterProjects("SofaComponentCommon_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_COMMON" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentCommon/SofaComponentCommon_test")

RegisterProjects("SofaComponentGeneral" OPTION "SOFA-LIB_COMPONENT_COMPONENT_GENERAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentGeneral")
#RegisterProjects("SofaComponentGeneral_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_GENERAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentGeneral/SofaComponentGeneral_test")

RegisterProjects("SofaComponentMisc" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMisc")
#RegisterProjects("SofaComponentMisc_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMisc/SofaComponentMisc_test")

RegisterProjects("SofaComponentBase" OPTION "SOFA-LIB_COMPONENT_COMPONENT_BASE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentBase")
#RegisterProjects("SofaComponentBase_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_BASE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentBase/SofaComponentBase_test")

RegisterProjects("SofaComponentMain" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MAIN" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMain")
#RegisterProjects("SofaComponentMain_test" OPTION "SOFA-LIB_COMPONENT_COMPONENT_MAIN" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaComponentMain/SofaComponentMain_test")

RegisterProjects("SofaBoundaryCondition" OPTION "SOFA-LIB_COMPONENT_BOUNDARY_CONDITION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBoundaryCondition")
RegisterProjects("SofaBoundaryCondition_test" OPTION "SOFA-LIB_COMPONENT_BOUNDARY_CONDITION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaBoundaryCondition/SofaBoundaryCondition_test")

RegisterProjects("SofaConstraint" OPTION "SOFA-LIB_COMPONENT_CONSTRAINT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaConstraint")
RegisterProjects("SofaConstraint_test" OPTION "SOFA-LIB_COMPONENT_CONSTRAINT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaConstraint/SofaConstraint_test")

RegisterProjects("SofaDeformable" OPTION "SOFA-LIB_COMPONENT_DEFORMABLE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeformable")
#RegisterProjects("SofaDeformable_test" OPTION "SOFA-LIB_COMPONENT_DEFORMABLE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDeformable/SofaDeformable_test")

RegisterProjects("SofaDenseSolver" OPTION "SOFA-LIB_COMPONENT_DENSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDenseSolver")
#RegisterProjects("SofaDenseSolver_test" OPTION "SOFA-LIB_COMPONENT_DENSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaDenseSolver/SofaDenseSolver_test")

RegisterProjects("SofaEigen2Solver" OPTION "SOFA-LIB_COMPONENT_EIGEN2_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEigen2Solver")
#RegisterProjects("SofaEigen2Solver_test" OPTION "SOFA-LIB_COMPONENT_EIGEN2_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEigen2Solver/SofaEigen2Solver_test")

RegisterProjects("SofaEngine" OPTION "SOFA-LIB_COMPONENT_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEngine")
RegisterProjects("SofaEngine_test" OPTION "SOFA-LIB_COMPONENT_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEngine/SofaEngine_test")

RegisterProjects("SofaEulerianFluid" OPTION "SOFA-LIB_COMPONENT_EULERIAN_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEulerianFluid")
#RegisterProjects("SofaEulerianFluid_test" OPTION "SOFA-LIB_COMPONENT_EULERIAN_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaEulerianFluid/SofaEulerianFluid_test")

RegisterProjects("SofaExplicitOdeSolver" OPTION "SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExplicitOdeSolver")
#RegisterProjects("SofaExplicitOdeSolver_test" OPTION "SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExplicitOdeSolver/SofaExplicitOdeSolver_test")

RegisterProjects("SofaExporter" OPTION "SOFA-LIB_COMPONENT_EXPORTER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExporter")
#RegisterProjects("SofaExporter_test" OPTION "SOFA-LIB_COMPONENT_EXPORTER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaExporter/SofaExporter_test")

RegisterProjects("SofaGraphComponent" OPTION "SOFA-LIB_COMPONENT_GRAPH_COMPONENT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGraphComponent")
#RegisterProjects("SofaGraphComponent_test" OPTION "SOFA-LIB_COMPONENT_GRAPH_COMPONENT" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaGraphComponent/SofaGraphComponent_test")

RegisterProjects("SofaHaptics" OPTION "SOFA-LIB_COMPONENT_HAPTICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaHaptics")
#RegisterProjects("SofaHaptics_test" OPTION "SOFA-LIB_COMPONENT_HAPTICS" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaHaptics/SofaHaptics_test")

RegisterProjects("SofaImplicitOdeSolver" OPTION "SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaImplicitOdeSolver")
#RegisterProjects("SofaImplicitOdeSolver_test" OPTION "SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaImplicitOdeSolver/SofaImplicitOdeSolver_test")

RegisterProjects("SofaLoader" OPTION "SOFA-LIB_COMPONENT_LOADER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaLoader")
RegisterProjects("SofaLoader_test" OPTION "SOFA-LIB_COMPONENT_LOADER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaLoader/SofaLoader_test")

RegisterProjects("SofaMeshCollision" OPTION "SOFA-LIB_COMPONENT_MESH_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMeshCollision")
#RegisterProjects("SofaMeshCollision_test" OPTION "SOFA-LIB_COMPONENT_MESH_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMeshCollision/SofaMeshCollision_test")

RegisterProjects("SofaMisc" OPTION "SOFA-LIB_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc")
#RegisterProjects("SofaMisc_test" OPTION "SOFA-LIB_COMPONENT_MISC" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMisc/SofaMisc_test")

RegisterProjects("SofaMiscCollision" OPTION "SOFA-LIB_COMPONENT_MISC_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscCollision")
#RegisterProjects("SofaMiscCollision_test" OPTION "SOFA-LIB_COMPONENT_MISC_COLLISION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscCollision/SofaMiscCollision_test")

RegisterProjects("SofaMiscEngine" OPTION "SOFA-LIB_COMPONENT_MISC_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscEngine")
#RegisterProjects("SofaMiscEngine_test" OPTION "SOFA-LIB_COMPONENT_MISC_ENGINE" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscEngine/SofaMiscEngine_test")

RegisterProjects("SofaMiscFem" OPTION "SOFA-LIB_COMPONENT_MISC_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscFem")
#RegisterProjects("SofaMiscFem_test" OPTION "SOFA-LIB_COMPONENT_MISC_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscFem/SofaMiscFem_test")

RegisterProjects("SofaMiscForceField" OPTION "SOFA-LIB_COMPONENT_MISC_FORCEFIELD" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscForceField")
#RegisterProjects("SofaMiscForceField_test" OPTION "SOFA-LIB_COMPONENT_MISC_FORCEFIELD" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscForceField/SofaMiscForceField_test")

RegisterProjects("SofaMiscMapping" OPTION "SOFA-LIB_COMPONENT_MISC_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscMapping")
#RegisterProjects("SofaMiscMapping_test" OPTION "SOFA-LIB_COMPONENT_MISC_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscMapping/SofaMiscMapping_test")

RegisterProjects("SofaMiscSolver" OPTION "SOFA-LIB_COMPONENT_MISC_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscSolver")
#RegisterProjects("SofaMiscSolver_test" OPTION "SOFA-LIB_COMPONENT_MISC_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscSolver/SofaMiscSolver_test")

RegisterProjects("SofaMiscTopology" OPTION "SOFA-LIB_COMPONENT_MISC_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscTopology")
#RegisterProjects("SofaMiscTopology_test" OPTION "SOFA-LIB_COMPONENT_MISC_TOPOLOGY" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaMiscTopology/SofaMiscTopology_test")

RegisterProjects("SofaNonUniformFem" OPTION "SOFA-LIB_COMPONENT_NON_UNIFORM_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaNonUniformFem")
#RegisterProjects("SofaNonUniformFem_test" OPTION "SOFA-LIB_COMPONENT_NON_UNIFORM_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaNonUniformFem/SofaNonUniformFem_test")

RegisterProjects("SofaObjectInteraction" OPTION "SOFA-LIB_COMPONENT_OBJECT_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaObjectInteraction")
#RegisterProjects("SofaObjectInteraction_test" OPTION "SOFA-LIB_COMPONENT_OBJECT_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaObjectInteraction/SofaObjectInteraction_test")

RegisterProjects("SofaOpenglVisual" OPTION "SOFA-LIB_COMPONENT_OPENGL_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaOpenglVisual") # NOT SOFA-MISC_NO_OPENGL
#RegisterProjects("SofaOpenglVisual_test" OPTION "SOFA-LIB_COMPONENT_OPENGL_VISUAL" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaOpenglVisual/SofaOpenglVisual_test")

RegisterProjects("SofaPardisoSolver" OPTION "SOFA-LIB_COMPONENT_PARDISO_SOLVER" COMPILE_DEFINITIONS SOFA_HAVE_PARDISO PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPardisoSolver") # previous option was : SOFA-LIB_PARDISO_SOLVER
#RegisterProjects("SofaPardisoSolver_test" OPTION "SOFA-LIB_COMPONENT_PARDISO_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPardisoSolver/SofaPardisoSolver_test")

RegisterProjects("SofaRigid" OPTION "SOFA-LIB_COMPONENT_RIGID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaRigid")
#RegisterProjects("SofaRigid_test" OPTION "SOFA-LIB_COMPONENT_RIGID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaRigid/SofaRigid_test")

RegisterProjects("SofaSimpleFem" OPTION "SOFA-LIB_COMPONENT_SIMPLE_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimpleFem")
#RegisterProjects("SofaSimpleFem_test" OPTION "SOFA-LIB_COMPONENT_SIMPLE_FEM" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSimpleFem/SofaSimpleFem_test")

RegisterProjects("SofaSparseSolver" OPTION "SOFA-LIB_COMPONENT_SPARSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSparseSolver")
#RegisterProjects("SofaSparseSolver_test" OPTION "SOFA-LIB_COMPONENT_SPARSE_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSparseSolver/SofaSparseSolver_test")

RegisterProjects("SofaPreconditioner" OPTION "SOFA-LIB_COMPONENT_PRECONDITIONER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPreconditioner")
#RegisterProjects("SofaPreconditioner_test" OPTION "SOFA-LIB_COMPONENT_PRECONDITIONER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaPreconditioner/SofaPreconditioner_test")

RegisterProjects("SofaSphFluid" OPTION "SOFA-LIB_COMPONENT_SPH_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSphFluid")
#RegisterProjects("SofaSphFluid_test" OPTION "SOFA-LIB_COMPONENT_SPH_FLUID" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaSphFluid/SofaSphFluid_test")

RegisterProjects("SofaTaucsSolver" OPTION "SOFA-LIB_COMPONENT_TAUCS_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTaucsSolver") #missing LIBS *= -lblas_win32_MT, etc..
#RegisterProjects("SofaTaucsSolver_test" OPTION "SOFA-LIB_COMPONENT_TAUCS_SOLVER" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTaucsSolver/SofaTaucsSolver_test")

RegisterProjects("SofaTopologyMapping" OPTION "SOFA-LIB_COMPONENT_TOPOLOGY_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTopologyMapping")
#RegisterProjects("SofaTopologyMapping_test" OPTION "SOFA-LIB_COMPONENT_TOPOLOGY_MAPPING" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaTopologyMapping/SofaTopologyMapping_test")

RegisterProjects("SofaUserInteraction" OPTION "SOFA-LIB_COMPONENT_USER_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaUserInteraction")
#RegisterProjects("SofaUserInteraction_test" OPTION "SOFA-LIB_COMPONENT_USER_INTERACTION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaUserInteraction/SofaUserInteraction_test")

RegisterProjects("SofaValidation" OPTION "SOFA-LIB_COMPONENT_VALIDATION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaValidation")
#RegisterProjects("SofaValidation_test" OPTION "SOFA-LIB_COMPONENT_VALIDATION" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaValidation/SofaValidation_test")

RegisterProjects("SofaVolumetricData" OPTION "SOFA-LIB_COMPONENT_VOLUMETRIC_DATA" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaVolumetricData")
#RegisterProjects("SofaVolumetricData_test" OPTION "SOFA-LIB_COMPONENT_VOLUMETRIC_DATA" PATH "${CMAKE_CURRENT_SOURCE_DIR}/SofaVolumetricData/SofaVolumetricData_test")
