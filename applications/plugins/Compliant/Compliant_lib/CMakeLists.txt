cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaCompliantPluginLibs REQUIRED)
include_directories(${SOFACOMPLIANTPLUGIN_INCLUDE_DIRS})

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
	UnilateralMask.h 
	RigidJointMapping.h 
	AssembledRigidRigidMapping.h 
	KKTSolver.h 
	LDLTSolver.h 
	MinresSolver.h 
	AssemblyVisitor.h 
	AssembledSystem.h 
	AssembledSolver.h 

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
	UnilateralMask.cpp 
	RigidJointMapping.cpp 
	AssembledRigidRigidMapping.cpp 
	LDLTSolver.cpp 
	MinresSolver.cpp 
	AssemblyVisitor.cpp 
	AssembledSystem.cpp 
	AssembledSolver.cpp 

	)

set(README_FILES

	Compliant.txt
	)	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_Compliant")
set(LINKER_DEPENDENCIES SofaGuiQt SofaEigen2Solver SofaFlexible amd lapack blas camd colamd cholmod)

include(${SOFA_CMAKE_DIR}/postProject.cmake)