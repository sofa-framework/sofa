cmake_minimum_required(VERSION 2.8.12)
project(Compliant)

set(COMPLIANT_MAJOR_VERSION 0)
set(COMPLIANT_MINOR_VERSION 0)
set(COMPLIANT_BUILD_VERSION 1)

set(COMPLIANT_VERSION ${COMPLIANT_MAJOR_VERSION}.${COMPLIANT_MINOR_VERSION}.${COMPLIANT_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(COMPLIANT_BUILD_LIB_DIR     ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(COMPLIANT_BUILD_BIN_DIR     ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(COMPLIANT_BUILD_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(COMPLIANT_BUILD_CMAKE_DIR   ${CMAKE_BINARY_DIR}/Compliant)

set(COMPLIANT_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(COMPLIANT_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(COMPLIANT_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(COMPLIANT_INSTALL_CMAKE_DIR   ${COMPLIANT_INSTALL_LIB_DIR}/cmake/Compliant )

set(HEADER_FILES
    initCompliant.h

    misc/CompliantAttachButtonSetting.h
    misc/RigidMass.h
    misc/CompliantSolverMerger.h
    misc/FailNode.h

    mapping/RigidJointMapping.h
	mapping/RigidJointMultiMapping.h
    mapping/AssembledRigidRigidMapping.h
    mapping/AssembledMapping.h
    mapping/AssembledMultiMapping.h
    mapping/DifferenceMapping.h
    mapping/DifferenceFromTargetMapping.h
    mapping/ContactMapping.h
    mapping/MaskMapping.h
    mapping/ProjectionMapping.h
    mapping/RigidComMultiMapping.h
    mapping/AffineMultiMapping.h
	mapping/PythonMultiMapping.h
    mapping/GearMapping.h

    numericalsolver/KKTSolver.h
    numericalsolver/EigenSparseSolver.h
    numericalsolver/EigenSparseSolver.inl
    numericalsolver/EigenSparseResponse.h
    numericalsolver/EigenSparseResponse.inl
    numericalsolver/MinresSolver.h
    numericalsolver/KrylovSolver.h
    numericalsolver/CgSolver.h
    numericalsolver/Response.h
    numericalsolver/DiagonalResponse.h
    numericalsolver/SequentialSolver.h
    numericalsolver/IterativeSolver.h
    numericalsolver/Benchmark.h
    numericalsolver/BenchmarkSolver.h
    numericalsolver/PreconditionedSolver.h
    numericalsolver/PreconditionedCgSolver.h
    numericalsolver/AnalysisSolver.h
    numericalsolver/LumpedResponse.h
    numericalsolver/BiCgStabSolver.h
    numericalsolver/NNCGSolver.h
	numericalsolver/SubKKT.h
    numericalsolver/SubKKT.inl
	numericalsolver/ModulusSolver.h

    preconditioner/BasePreconditioner.h
    preconditioner/CompliantLDLTPreconditioner.h
    preconditioner/IncompleteCholeskyPreconditioner.h
    preconditioner/CompliantJacobiPreconditioner.h

    assembly/AssemblyHelper.h
    assembly/AssemblyVisitor.h
    assembly/AssembledSystem.h

    odesolver/CompliantImplicitSolver.h
    odesolver/ConstantCompliantImplicitSolver.h
    odesolver/CompliantNLImplicitSolver.h
    odesolver/CompliantPostStabilizationAnimationLoop.h
    odesolver/CompliantModalAnalysis.h

    compliance/UniformCompliance.h
    compliance/UniformCompliance.inl
    compliance/DiagonalCompliance.h
    compliance/DiagonalCompliance.inl
    compliance/DampingCompliance.h

    constraint/Constraint.h
    constraint/CoulombConstraint.h
    constraint/UnilateralConstraint.h
    constraint/ResistanceConstraint.h
    constraint/ClosureConstraint.h

    constraint/BaseConstraintValue.h
    constraint/Stabilization.h
    constraint/ConstraintValue.h
    constraint/Restitution.h
    constraint/DampingValue.h
    constraint/OffsettedConstraintValue.h
    constraint/HolonomicConstraintValue.h
    constraint/VelocityConstraintValue.h

    contact/CompliantContact.h
    contact/FrictionCompliantContact.h
    contact/BaseContact.h
    contact/PenalityCompliantContact.h

    controller/CompliantSleepController.h
    
    utils/basis.h
    utils/cast.h
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
    utils/thread_variable.h
	utils/anderson.h
	utils/sub_kkt.h
	utils/sub_kkt.inl
	
    utils/cg.h
    utils/preconditionedcg.h
    utils/minres.h
    utils/bicgstab.h
)

set(SOURCE_FILES
    initCompliant.cpp

    misc/CompliantAttachButtonSetting.cpp
    misc/CompliantSolverMerger.cpp
    misc/RigidMass.cpp
    misc/FailNode.cpp
	misc/python.cpp

    compliance/UniformCompliance.cpp
    compliance/DiagonalCompliance.cpp
    compliance/DampingCompliance.cpp

    mapping/RigidJointMapping.cpp
    mapping/RigidJointMultiMapping.cpp
    mapping/AssembledRigidRigidMapping.cpp
    mapping/DifferenceMapping.cpp
    mapping/DifferenceFromTargetMapping.cpp
    mapping/ContactMapping.cpp
    mapping/MaskMapping.cpp 
    mapping/ProjectionMapping.cpp
    mapping/RigidComMultiMapping.cpp
    mapping/AffineMultiMapping.cpp
	mapping/PythonMultiMapping.cpp
    mapping/GearMapping.cpp

    numericalsolver/EigenSparseSolver.cpp
    numericalsolver/EigenSparseResponse.cpp
    numericalsolver/MinresSolver.cpp
    numericalsolver/KrylovSolver.cpp
    numericalsolver/CgSolver.cpp
    numericalsolver/DiagonalResponse.cpp
    numericalsolver/SequentialSolver.cpp
    numericalsolver/IterativeSolver.cpp
    numericalsolver/Benchmark.cpp
    numericalsolver/BenchmarkSolver.cpp
    numericalsolver/PreconditionedSolver.cpp
    numericalsolver/PreconditionedCgSolver.cpp
    numericalsolver/AnalysisSolver.cpp
    numericalsolver/LumpedResponse.cpp
    numericalsolver/BiCgStabSolver.cpp
    numericalsolver/NNCGSolver.cpp
	numericalsolver/SubKKT.cpp
	numericalsolver/ModulusSolver.cpp

    preconditioner/CompliantLDLTPreconditioner.cpp
    preconditioner/IncompleteCholeskyPreconditioner.cpp
    preconditioner/CompliantJacobiPreconditioner.cpp

    assembly/AssemblyVisitor.cpp
    assembly/AssembledSystem.cpp

    odesolver/CompliantImplicitSolver.cpp
    odesolver/ConstantCompliantImplicitSolver.cpp
    odesolver/CompliantNLImplicitSolver.cpp
    odesolver/CompliantPostStabilizationAnimationLoop.cpp
    odesolver/CompliantModalAnalysis.cpp

    constraint/CoulombConstraint.cpp
    constraint/UnilateralConstraint.cpp
    constraint/ResistanceConstraint.cpp
    constraint/ClosureConstraint.cpp

    constraint/ConstraintValue.cpp
    constraint/Stabilization.cpp
    constraint/Restitution.cpp
    constraint/DampingValue.cpp
    constraint/OffsettedConstraintValue.cpp
    constraint/HolonomicConstraintValue.cpp
    constraint/VelocityConstraintValue.cpp

    contact/CompliantContact.cpp
    contact/FrictionCompliantContact.cpp
    contact/PenalityCompliantContact.cpp
	
	controller/CompliantSleepController.cpp

   	utils/anderson.cpp
   	utils/sub_kkt.cpp)

set(README_FILES Compliant.txt)

# include python files
include(./python/python.cmake)	


find_package(SofaMisc REQUIRED)

set(COMPLIANT_DEPENDENCY_INCLUDE_DIR ${SofaMisc_INCLUDE_DIRS})

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Compliant.h.in
                ${CMAKE_CURRENT_BINARY_DIR}/Compliant/Compliant.h)
install( FILES ${CMAKE_CURRENT_BINARY_DIR}/Compliant/Compliant.h
         DESTINATION "${COMPLIANT_INSTALL_INCLUDE_DIR}/Compliant")
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaSimulationGraph)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>") 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${COMPLIANT_DEPENDENCY_INCLUDE_DIR}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${COMPLIANT_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_Compliant")

install(TARGETS ${PROJECT_NAME}
        COMPONENT Compliant_libraries
        EXPORT  CompliantTargets
        RUNTIME DESTINATION "${COMPLIANT_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${COMPLIANT_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${COMPLIANT_INSTALL_LIB_DIR}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION "${COMPLIANT_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}/${DIR}")
endforeach(HEADER HEADERS)

find_package(SofaTest QUIET)
if(SofaTest_FOUND)
    add_subdirectory(Compliant_test)
endif()

include(CMakePackageConfigHelpers)
# CompliantConfigVersion.cmake file (same for build tree and intall)
write_basic_package_version_file(${COMPLIANT_BUILD_CMAKE_DIR}/CompliantConfigVersion.cmake
                                 VERSION ${COMPLIANT_VERSION}
                                 COMPATIBILITY AnyNewerVersion)

install(FILES ${COMPLIANT_BUILD_CMAKE_DIR}/CompliantConfigVersion.cmake
        DESTINATION ${COMPLIANT_INSTALL_CMAKE_DIR})

# linking with boost-system is needed
# that means an installed version of boost is mandatory (sadly)
# I loved the header-only boost

# CompliantConfig.cmake (build tree)
set(COMPLIANT_INCLUDE_DIR "${COMPLIANT_BUILD_INCLUDE_DIR}")
configure_package_config_file(CompliantConfig.cmake.in
                              ${COMPLIANT_BUILD_CMAKE_DIR}/CompliantConfig.cmake
                              INSTALL_DESTINATION ${COMPLIANT_BUILD_CMAKE_DIR}
                              PATH_VARS COMPLIANT_INCLUDE_DIR)

# CompliantConfig.cmake (installed)
set(COMPLIANT_INCLUDE_DIR "${COMPLIANT_INSTALL_INCLUDE_DIR}")
configure_package_config_file(CompliantConfig.cmake.in
                              ${COMPLIANT_BUILD_CMAKE_DIR}/CompliantConfigInstall.cmake
                              INSTALL_DESTINATION ${COMPLIANT_INSTALL_CMAKE_DIR}
                              PATH_VARS COMPLIANT_INCLUDE_DIR)

install(FILES ${COMPLIANT_BUILD_CMAKE_DIR}/CompliantConfigInstall.cmake
        DESTINATION ${COMPLIANT_INSTALL_CMAKE_DIR}
        RENAME CompliantConfig.cmake)

# CompliantTargets.cmake (installed)
install(EXPORT CompliantTargets
        DESTINATION ${COMPLIANT_INSTALL_CMAKE_DIR})
