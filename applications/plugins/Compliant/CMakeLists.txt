include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	initCompliant.h

    misc/CompliantAttachButtonSetting.h
    misc/RigidMass.h
    misc/CompliantSolverMerger.h

    mapping/RigidJointMapping.h
    mapping/AssembledRigidRigidMapping.h
    mapping/AssembledMapping.h
    mapping/AssembledMultiMapping.h
    mapping/DifferenceMapping.h
    mapping/ContactMapping.h
    mapping/MaskMapping.h


    numericalsolver/KKTSolver.h
    numericalsolver/LDLTSolver.h
    numericalsolver/MinresSolver.h
    numericalsolver/KrylovSolver.h
    numericalsolver/CgSolver.h
    numericalsolver/Response.h
    numericalsolver/LDLTResponse.h
    numericalsolver/DiagonalResponse.h
    numericalsolver/SequentialSolver.h

    assembly/AssemblyHelper.h
    assembly/AssemblyVisitor.h
    assembly/AssembledSystem.h

    odesolver/AssembledSolver.h

    compliance/UniformCompliance.h
    compliance/UniformCompliance.inl
    compliance/DiagonalCompliance.h
    compliance/DiagonalCompliance.inl
    compliance/DampingCompliance.h

    constraint/Constraint.h
    constraint/CoulombConstraint.h
    constraint/UnilateralConstraint.h

    constraint/BaseConstraintValue.h
    constraint/Stabilization.h
    constraint/ConstraintValue.h
    constraint/Restitution.h
    constraint/DampingValue.h

    contact/CompliantContact.h
    contact/FrictionCompliantContact.h
    contact/BaseContact.h
	
    utils/basis.h
    utils/bench.h
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
    utils/schur.h
    utils/scoped.h
    utils/se3.h
    utils/sparse.h
    utils/thread_local.h
	)

set(SOURCE_FILES

  initCompliant.cpp

    misc/CompliantAttachButtonSetting.cpp
    misc/CompliantSolverMerger.cpp
    misc/RigidMass.cpp

    compliance/UniformCompliance.cpp
    compliance/DiagonalCompliance.cpp
    compliance/DampingCompliance.cpp

    mapping/RigidJointMapping.cpp
    mapping/RigidJointMultiMapping.cpp
    mapping/AssembledRigidRigidMapping.cpp
    mapping/DifferenceMapping.cpp
    mapping/ContactMapping.cpp
    mapping/MaskMapping.cpp 

    numericalsolver/LDLTSolver.cpp
    numericalsolver/MinresSolver.cpp
    numericalsolver/KrylovSolver.cpp
    numericalsolver/CgSolver.cpp
    numericalsolver/LDLTResponse.cpp
    numericalsolver/DiagonalResponse.cpp
    numericalsolver/SequentialSolver.cpp

    assembly/AssemblyVisitor.cpp
    assembly/AssembledSystem.cpp

    odesolver/AssembledSolver.cpp

    constraint/CoulombConstraint.cpp
    constraint/UnilateralConstraint.cpp

    constraint/ConstraintValue.cpp
    constraint/Stabilization.cpp
    constraint/Restitution.cpp
    constraint/DampingValue.cpp

    contact/CompliantContact.cpp
    contact/FrictionCompliantContact.cpp
	)

set(README_FILES

	Compliant.txt
	)

if(WIN32)
	include_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Include")
	link_directories("${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib")
else()
	include_directories("/usr/include/suitesparse/")
endif()
	
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_Compliant")
AddLinkerDependencies(SofaEigen2Solver SofaUserInteraction SofaComponentMisc SofaHelper SofaSimulationGraph)
if(NOT APPLE)
	AddLinkerDependencies(amd lapack blas camd colamd cholmod)
endif()

# WIN32: copy the needed dlls from the lib dir to the Sofa bin dir
if(WIN32)
	file(GLOB sharedObjects "${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib/*.dll")
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
	endforeach()
endif()
	
include(${SOFA_CMAKE_DIR}/postProject.cmake)
include(custom.cmake OPTIONAL)
