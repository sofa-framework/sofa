include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    gui/qt/QMouseOperations.h
	)

set(SOURCE_FILES

    pim/ProgressiveScaling.cpp 
    pim/ComputeMeshIntersection.cpp 
    pim/Parameters.cpp 
    pim/TransformPlaneConstraint.cpp 
    pim/SculptBodyPerformer.cpp
	
	gui/MouseOperations.cpp
	
	gui/qt/QMouseOperations.cpp
	
	init/initPim.cpp
	)
	
UseQt()
SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})

set(HEADER_FILES

	${HEADER_FILES}

    pim/SculptBodyPerformer.h 
    pim/SculptBodyPerformer.inl 
    pim/ProgressiveScaling.h 
    pim/ProgressiveScaling.inl 
    pim/ComputeMeshIntersection.h 
    pim/ComputeMeshIntersection.inl 
    pim/Parameters.h 
    pim/TransformPlaneConstraint.h
	
	gui/MouseOperations.h
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_PIM" "SOFA_BUILD_PIMGUI")
AddLinkerDependencies(SofaSimulationCommon SofaComponentLinearSolver SofaComponentOdeSolver SofaComponentMass SofaComponentForcefield SofaComponentCollision SofaComponentConstraint SofaComponentMisc SofaComponentBase SofaComponentMapping SofaComponentEngine SofaComponentVisualModel SofaGui SofaGuiMain) # TriangularMeshRefiner / SofaCGALPlugin

include(${SOFA_CMAKE_DIR}/postProject.cmake)