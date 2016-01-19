cmake_minimum_required(VERSION 2.8.12)
project(Compliant)

set(COMPLIANT_MAJOR_VERSION 0)
set(COMPLIANT_MINOR_VERSION 1)
set(COMPLIANT_VERSION ${COMPLIANT_MAJOR_VERSION}.${COMPLIANT_MINOR_VERSION})

set(HEADER_FILES
    assembly/AssembledSystem.h
    assembly/AssemblyHelper.h
    assembly/AssemblyVisitor.h
    compliance/DampingCompliance.h
    compliance/DiagonalCompliance.h
    compliance/DiagonalCompliance.inl
    compliance/UniformCompliance.h
    compliance/UniformCompliance.inl
    compliance/FullCompliance.h
    compliance/FullCompliance.inl
    constraint/BaseConstraintValue.h
    constraint/ClosureConstraint.h
    constraint/Constraint.h
    constraint/ConstraintValue.h
    constraint/CoulombConstraint.h
    constraint/CoulombConstraint.inl
    constraint/DampingValue.h
    constraint/HolonomicConstraintValue.h
    constraint/OffsettedConstraintValue.h
    constraint/ResistanceConstraint.h
    constraint/Restitution.h
    constraint/Stabilization.h
    constraint/UnilateralConstraint.h
    constraint/VelocityConstraintValue.h
    constraint/BaumgarteStabilization.h
    contact/BaseContact.h
    contact/CompliantContact.h
    contact/FrictionCompliantContact.h
    contact/PenalityCompliantContact.h
    controller/CompliantSleepController.h
    forcefield/PotentialEnergy.h
    forcefield/UniformLinearPotentialEnergy.h
    mapping/AffineMultiMapping.h
    mapping/AssembledMapping.h
    mapping/AssembledMultiMapping.h
    mapping/AssembledRigidRigidMapping.h
    mapping/ContactMapping.h
    mapping/DifferenceFromTargetMapping.h
    mapping/DifferenceMapping.h
    mapping/GearMapping.h
    mapping/MaskMapping.h
    mapping/PairingMultiMapping.h
    mapping/ProjectionMapping.h
    mapping/PythonMultiMapping.h
    mapping/QuadraticMapping.h
    mapping/RigidComMultiMapping.h
    mapping/RigidJointMapping.h
    mapping/RigidJointMultiMapping.h
    misc/CompliantAttachButtonSetting.h
    misc/CompliantSolverMerger.h
    misc/FailNode.h
    misc/RigidMass.h
    numericalsolver/AnalysisSolver.h
    numericalsolver/Benchmark.h
    numericalsolver/BenchmarkSolver.h
    numericalsolver/BiCgStabSolver.h
    numericalsolver/CgSolver.h
    numericalsolver/DiagonalResponse.h
    numericalsolver/EigenSparseResponse.h
    numericalsolver/EigenSparseResponse.inl
    numericalsolver/EigenSparseSolver.h
    numericalsolver/EigenSparseSolver.inl
    numericalsolver/IterativeSolver.h
    numericalsolver/KKTSolver.h
    numericalsolver/KrylovSolver.h
    numericalsolver/LumpedResponse.h
    numericalsolver/MinresSolver.h
    numericalsolver/ModulusSolver.h
    numericalsolver/NNCGSolver.h
    numericalsolver/PreconditionedCgSolver.h
    numericalsolver/PreconditionedSolver.h
    numericalsolver/Response.h
    numericalsolver/SequentialSolver.h
    numericalsolver/SubKKT.h
    numericalsolver/SubKKT.inl
    odesolver/CompliantImplicitSolver.h
    odesolver/CompliantNLImplicitSolver.h
    odesolver/CompliantPostStabilizationAnimationLoop.h
    odesolver/CompliantPseudoStaticSolver.h
    odesolver/CompliantPseudoStaticSolver.inl
    odesolver/CompliantStaticSolver.h
    odesolver/ConstantCompliantImplicitSolver.h
    preconditioner/BasePreconditioner.h
    preconditioner/CompliantJacobiPreconditioner.h
    preconditioner/CompliantLDLTPreconditioner.h
    preconditioner/IncompleteCholeskyPreconditioner.h
    utils/anderson.h
    utils/basis.h
    utils/bicgstab.h
    utils/cg.h
    utils/cone.h
    utils/edit.h
    utils/find.h
    utils/force_assert.h
    utils/graph.h
    utils/kkt.h
    utils/krylov.h
    utils/map.h
    utils/minres.h
    utils/nan.h
    utils/pair.h
    utils/preconditionedcg.h
    utils/schur.h
    utils/scoped.h
    utils/se3.h
    utils/sparse.h
    utils/sub_kkt.h
    utils/sub_kkt.inl
    utils/thread_variable.h
)

set(SOURCE_FILES
    assembly/AssembledSystem.cpp
    assembly/AssemblyVisitor.cpp
    compliance/DampingCompliance.cpp
    compliance/DiagonalCompliance.cpp
    compliance/UniformCompliance.cpp
    compliance/FullCompliance.cpp
    constraint/ClosureConstraint.cpp
    constraint/Constraint.cpp
    constraint/ConstraintValue.cpp
    constraint/CoulombConstraint.cpp
    constraint/DampingValue.cpp
    constraint/HolonomicConstraintValue.cpp
    constraint/OffsettedConstraintValue.cpp
    constraint/ResistanceConstraint.cpp
    constraint/Restitution.cpp
    constraint/Stabilization.cpp
    constraint/UnilateralConstraint.cpp
    constraint/VelocityConstraintValue.cpp
    constraint/BaumgarteStabilization.cpp
    contact/CompliantContact.cpp
    contact/FrictionCompliantContact.cpp
    contact/PenalityCompliantContact.cpp
    controller/CompliantSleepController.cpp
    forcefield/PotentialEnergy.cpp
    forcefield/UniformLinearPotentialEnergy.cpp
    initCompliant.cpp
    mapping/AffineMultiMapping.cpp
    mapping/AssembledRigidRigidMapping.cpp
    mapping/ContactMapping.cpp
    mapping/DifferenceFromTargetMapping.cpp
    mapping/DifferenceMapping.cpp
    mapping/GearMapping.cpp
    mapping/MaskMapping.cpp
    mapping/PairingMultiMapping.cpp
    mapping/ProjectionMapping.cpp
    mapping/PythonMultiMapping.cpp
    mapping/QuadraticMapping.cpp
    mapping/RigidComMultiMapping.cpp
    mapping/RigidJointMapping.cpp
    mapping/RigidJointMultiMapping.cpp
    misc/CompliantAttachButtonSetting.cpp
    misc/CompliantSolverMerger.cpp
    misc/FailNode.cpp
    misc/RigidMass.cpp
    misc/python.cpp
    numericalsolver/AnalysisSolver.cpp
    numericalsolver/Benchmark.cpp
    numericalsolver/BenchmarkSolver.cpp
    numericalsolver/BiCgStabSolver.cpp
    numericalsolver/CgSolver.cpp
    numericalsolver/DiagonalResponse.cpp
    numericalsolver/EigenSparseResponse.cpp
    numericalsolver/EigenSparseSolver.cpp
    numericalsolver/IterativeSolver.cpp
    numericalsolver/KrylovSolver.cpp
    numericalsolver/LumpedResponse.cpp
    numericalsolver/MinresSolver.cpp
    numericalsolver/ModulusSolver.cpp
    numericalsolver/NNCGSolver.cpp
    numericalsolver/PreconditionedCgSolver.cpp
    numericalsolver/PreconditionedSolver.cpp
    numericalsolver/SequentialSolver.cpp
    numericalsolver/SubKKT.cpp
    odesolver/CompliantImplicitSolver.cpp
    odesolver/CompliantNLImplicitSolver.cpp
    odesolver/CompliantPostStabilizationAnimationLoop.cpp
    odesolver/CompliantPseudoStaticSolver.cpp
    odesolver/CompliantStaticSolver.cpp
    odesolver/ConstantCompliantImplicitSolver.cpp
    preconditioner/CompliantJacobiPreconditioner.cpp
    preconditioner/CompliantLDLTPreconditioner.cpp
    preconditioner/IncompleteCholeskyPreconditioner.cpp
    utils/anderson.cpp
    utils/sub_kkt.cpp
)

set(PYTHON_FILES
    python/Compliant/Control.py
    python/Compliant/Frame.py
    python/Compliant/Quaternion.py
    python/Compliant/Rigid.py
    python/Compliant/Script.py
    python/Compliant/StructuralAPI.py
    python/Compliant/Tools.py
    python/Compliant/Vec.py
    python/Compliant/__init__.py
    python/Compliant/future.py
    python/Compliant/sml.py
)

set(README_FILES Compliant.txt)

find_package(SofaMisc REQUIRED)

## Compliant/config.h
configure_file(config.h.in "${CMAKE_BINARY_DIR}/include/Compliant/config.h")
install(FILES "${CMAKE_BINARY_DIR}/include/Compliant/config.h" DESTINATION "include/Compliant")

# Config files and install rules for pythons scripts
sofa_set_python_directory(${PROJECT_NAME} "python")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaSimulationGraph)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${COMPLIANT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Compliant")

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(Compliant_test)
    endif()
endif()

find_package(SofaGui QUIET)
if(SofaGui_FOUND AND SOFA_GUI_QT)
    add_subdirectory(Compliant_gui)
endif()

## Install rules for the library; CMake package configurations files
sofa_create_package(Compliant ${COMPLIANT_VERSION} Compliant Compliant)

## Install rules for the resources
install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

## Hand-crafted install rules for the headers, to preserve the directory structure
foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION "include/${PROJECT_NAME}/${DIR}")
endforeach(HEADER HEADERS)
