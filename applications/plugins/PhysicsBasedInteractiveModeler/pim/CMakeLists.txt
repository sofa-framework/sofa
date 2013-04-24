cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
find_package( SofaPimLibs REQUIRED)
include_directories(${SOFAPIM_INCLUDE_DIRS})

set(HEADER_FILES

    SculptBodyPerformer.h 
    SculptBodyPerformer.inl 
    ProgressiveScaling.h 
    ProgressiveScaling.inl 
    ComputeMeshIntersection.h 
    ComputeMeshIntersection.inl 
    Parameters.h 
    TransformPlaneConstraint.h 
    EventManager.inl 
    EventManager.h

	)

set(SOURCE_FILES

    ProgressiveScaling.cpp 
    ComputeMeshIntersection.cpp 
    Parameters.cpp 
    TransformPlaneConstraint.cpp 
    EventManager.cpp 
    SculptBodyPerformer.cpp

	)	

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_PIM")
set(LINKER_DEPENDENCIES SofaSimulationCommon SofaComponentLinearSolver SofaComponentOdeSolver SofaComponentMass SofaComponentForcefield SofaComponentCollision SofaComponentConstraint SofaComponentMisc SofaComponentBase SofaComponentMapping SofaComponentEngine SofaComponentVisualModel SofaCGALPlugin SofaGui SofaGuiMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)