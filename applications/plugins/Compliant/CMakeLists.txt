include(${SOFA_CMAKE_DIR}/preProject.cmake)

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
	mapping/PairingMultiMapping.h
	mapping/QuadraticMapping.h

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
    odesolver/CompliantPseudoStaticSolver.h
    odesolver/CompliantPostStabilizationAnimationLoop.h
	odesolver/CompliantStaticSolver.h

    forcefield/PotentialEnergy.h
    forcefield/UniformLinearPotentialEnergy.h
	
    compliance/UniformCompliance.h
    compliance/UniformCompliance.inl
    compliance/DiagonalCompliance.h
    compliance/DiagonalCompliance.inl
    compliance/DampingCompliance.h

    constraint/Constraint.h
    constraint/CoulombConstraint.h
	constraint/CoulombConstraint.inl
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
    utils/cone.h
    utils/edit.h
    utils/find.h
    utils/force_assert.h
    utils/graph.h
    utils/kkt.h
    utils/krylov.h
    utils/map.h
    utils/nan.h
    utils/pair.h
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

    forcefield/PotentialEnergy.cpp
    forcefield/UniformLinearPotentialEnergy.cpp
	
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
	mapping/PairingMultiMapping.cpp
	mapping/QuadraticMapping.cpp

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
    odesolver/CompliantPseudoStaticSolver.cpp
    odesolver/CompliantPostStabilizationAnimationLoop.cpp
	odesolver/CompliantStaticSolver.cpp

	constraint/Constraint.cpp
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

	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES} ${PYTHON_FILES})

AddCompilerDefinitions("SOFA_BUILD_Compliant")
AddLinkerDependencies( SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaSimulationGraph )




#### BOOST ######

# Compliant is forced to use a header-only Boost
# But sadely, some configs (e.g. gcc4.6 only in debug)
# still require to link with a compiled Boost system lib.
# So if this lib is installed we always link with it.
# Otherwise we pray the header-only compilation works.
# Note at this level the system component has already been looked
# for in externals.cmake, so no need to look for it again
# (and with a risk of mixing several boost versions)
if( Boost_SYSTEM_LIBRARY )
    AddLinkerDependencies( ${Boost_SYSTEM_LIBRARY} )
endif()


if(LINUX OR UNIX AND NOT APPLE)
    AddLinkerDependencies( rt ) # clock_gettime
endif()






include(${SOFA_CMAKE_DIR}/postProject.cmake)
include(custom.cmake OPTIONAL)

