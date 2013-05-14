include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	initCompliant.h
    CompliantAttachButtonSetting.h
    CompliantAttachPerformer.h
    CompliantAttachPerformer.inl
    UniformCompliance.h
    UniformCompliance.inl
    ComplianceSolver.h
	DampingCompliance.h
	RigidMass.h
#	UnilateralMask.h
	SolverFlags.h
	RigidJointMapping.h
	AssembledRigidRigidMapping.h
	KKTSolver.h
	LDLTSolver.h
	MinresSolver.h
	AssemblyVisitor.h
	AssembledSystem.h
	AssembledSolver.h
	Projector.h
	)

set(SOURCE_FILES

	initCompliant.cpp
    CompliantAttachButtonSetting.cpp
    CompliantAttachPerformer.cpp
    UniformCompliance.cpp
    DiagonalCompliance.cpp
    ComplianceSolver.cpp
	DampingCompliance.cpp
	RigidMass.cpp
#	UnilateralMask.cpp
	SolverFlags.cpp
	RigidJointMapping.cpp
	AssembledRigidRigidMapping.cpp
	LDLTSolver.cpp
	MinresSolver.cpp
	AssemblyVisitor.cpp
	AssembledSystem.cpp
	AssembledSolver.cpp
	Projector.cpp
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

set(COMPILER_DEFINES "SOFA_BUILD_Compliant")
set(LINKER_DEPENDENCIES Flexible SofaGuiQt SofaEigen2Solver SofaSimulationGraph amd lapack blas camd colamd cholmod)

# WIN32: copy the needed dlls from the lib dir to the Sofa bin dir
if(WIN32)
	file(GLOB sharedObjects "${SOFA_EXTLIBS_DIR}/SuiteSparse/cholmod/Lib/*.dll")
	foreach(sharedObject ${sharedObjects})
		file(COPY ${sharedObject} DESTINATION "${SOFA_BIN_DIR}")
	endforeach()
endif()
	
include(${SOFA_CMAKE_DIR}/postProject.cmake)