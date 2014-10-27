include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    initCompliant.h

    misc/CompliantAttachButtonSetting.h
    misc/RigidMass.h
    misc/CompliantSolverMerger.h
    misc/FailNode.h

    mapping/RigidJointMapping.h
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

    numericalsolver/KKTSolver.h
    numericalsolver/LDLTSolver.h
    numericalsolver/MinresSolver.h
    numericalsolver/KrylovSolver.h
    numericalsolver/CgSolver.h
    numericalsolver/Response.h
    numericalsolver/LDLTResponse.h
    numericalsolver/DiagonalResponse.h
    numericalsolver/SequentialSolver.h
    numericalsolver/IterativeSolver.h
    numericalsolver/Benchmark.h
    numericalsolver/BenchmarkSolver.h
    numericalsolver/PreconditionedSolver.h
    numericalsolver/PreconditionedCgSolver.h

    preconditioner/BasePreconditioner.h
    preconditioner/CompliantLDLTPreconditioner.h
    #preconditioner/IncompleteCholeskyPreconditioner.h
    preconditioner/CompliantJacobiPreconditioner.h

    assembly/AssemblyHelper.h
    assembly/AssemblyVisitor.h
    assembly/AssembledSystem.h

    odesolver/CompliantImplicitSolver.h
    odesolver/ConstantCompliantImplicitSolver.h
    odesolver/CompliantNLImplicitSolver.h
    odesolver/CompliantPostStabilizationAnimationLoop.h

    compliance/UniformCompliance.h
    compliance/UniformCompliance.inl
    compliance/DiagonalCompliance.h
    compliance/DiagonalCompliance.inl
    compliance/DampingCompliance.h

    constraint/Constraint.h
    constraint/CoulombConstraint.h
    constraint/UnilateralConstraint.h
    constraint/ResistanceConstraint.h

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
    utils/pair.h
    utils/preconditionedcg.h
    utils/schur.h
    utils/scoped.h
    utils/se3.h
    utils/sparse.h
    utils/thread_local.h)

set(SOURCE_FILES
    initCompliant.cpp

    misc/CompliantAttachButtonSetting.cpp
    misc/CompliantSolverMerger.cpp
    misc/RigidMass.cpp
    misc/FailNode.cpp

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

    numericalsolver/LDLTSolver.cpp
    numericalsolver/MinresSolver.cpp
    numericalsolver/KrylovSolver.cpp
    numericalsolver/CgSolver.cpp
    numericalsolver/LDLTResponse.cpp
    numericalsolver/DiagonalResponse.cpp
    numericalsolver/SequentialSolver.cpp
    numericalsolver/IterativeSolver.cpp
    numericalsolver/Benchmark.cpp
    numericalsolver/BenchmarkSolver.cpp
    numericalsolver/PreconditionedSolver.cpp
    numericalsolver/PreconditionedCgSolver.cpp

    preconditioner/CompliantLDLTPreconditioner.cpp
    #preconditioner/IncompleteCholeskyPreconditioner.cpp
    preconditioner/CompliantJacobiPreconditioner.cpp

    assembly/AssemblyVisitor.cpp
    assembly/AssembledSystem.cpp

    odesolver/CompliantImplicitSolver.cpp
    odesolver/ConstantCompliantImplicitSolver.cpp
    odesolver/CompliantNLImplicitSolver.cpp
    odesolver/CompliantPostStabilizationAnimationLoop.cpp

    constraint/CoulombConstraint.cpp
    constraint/UnilateralConstraint.cpp
    constraint/ResistanceConstraint.cpp

    constraint/ConstraintValue.cpp
    constraint/Stabilization.cpp
    constraint/Restitution.cpp
    constraint/DampingValue.cpp
    constraint/OffsettedConstraintValue.cpp
    constraint/HolonomicConstraintValue.cpp
    constraint/VelocityConstraintValue.cpp

    contact/CompliantContact.cpp
    contact/FrictionCompliantContact.cpp
    contact/PenalityCompliantContact.cpp)

set(README_FILES Compliant.txt)

# include python files
include(./python/python.cmake)	
	
#if(WIN32)
#	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
#	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
#else()
#	include_directories("/usr/include/suitesparse/")
#endif()
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})

AddCompilerDefinitions("SOFA_BUILD_Compliant")
AddLinkerDependencies(SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaHelper SofaSimulationGraph)
AddSourceDependencies( GenerateRigid )

# is this really useful?
#if(NOT APPLE)
#	AddLinkerDependencies(amd lapack blas camd colamd cholmod)
#endif()

# WIN32: copy the needed dlls from the lib dir to the Sofa bin dir
#if(WIN32)
#	file(GLOB sharedObjects "${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib/*.dll")
#	foreach(sharedObject ${sharedObjects})
#		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
#	endforeach()
#endif()







#### BOOST ######

# linking with boost-system is needed
# that means an installed version of boost is mandatory (sadly)
# I loved the header-only boost

# max: hopefully this is not needed anymore

# if(NOT SOFA-EXTERNAL_BOOST)
#    message(FATAL_ERROR "Compliant plugin needs a compiled Boost library >= 1.48. Please modify the SOFA-EXTERNAL_BOOST_PATH option to fix it...")
# endif()
# find_package(Boost COMPONENTS system REQUIRED)
# AddLinkerDependencies( ${Boost_SYSTEM_LIBRARY} )


if(LINUX OR UNIX AND NOT APPLE)
    AddLinkerDependencies( rt ) # clock_gettime
endif()






include(${SOFA_CMAKE_DIR}/postProject.cmake)
include(custom.cmake OPTIONAL)

