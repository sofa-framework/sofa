include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

	common.h
	Node.h
	AnimateBeginEvent.h
	AnimateEndEvent.h
	AnimateVisitor.h   
	BehaviorUpdatePositionVisitor.h
	CactusStackStorage.h  
	ClassSystem.h  
	CleanupVisitor.h
	CollisionAnimationLoop.h
	CollisionBeginEvent.h
	CollisionEndEvent.h
	CollisionVisitor.h  
	Colors.h
	CopyAspectVisitor.h
	DeactivatedNodeVisitor.h
	DefaultAnimationLoop.h
	DefaultVisualManagerLoop.h
	DeleteVisitor.h
	ExportGnuplotVisitor.h
	ExportOBJVisitor.h
	FindByTypeVisitor.h
	InitVisitor.h  
	IntegrateBeginEvent.h
	IntegrateEndEvent.h
	LocalStorage.h
	MechanicalComputeEnergyVisitor.h
    MechanicalGetMomentumVisitor.h
	MechanicalMatrixVisitor.h
	 MechanicalOperations.h
	MechanicalVPrintVisitor.h
	MechanicalVisitor.h
	MutationListener.h
	ParallelVisitorScheduler.h
	PauseEvent.h
	PipelineImpl.h
	PositionEvent.h
	PrintVisitor.h
	PropagateEventVisitor.h
	ReleaseAspectVisitor.h
	ResetVisitor.h
	Simulation.h
	SolveVisitor.h
	StateChangeVisitor.h
	TopologyChangeVisitor.h
	TransformationVisitor.h
	UpdateBoundingBoxVisitor.h
	UpdateContextVisitor.h
	UpdateLinksVisitor.h
	UpdateMappingEndEvent.h
	UpdateMappingVisitor.h
	VectorOperations.h
	VelocityThresholdVisitor.h
	Visitor.h
	VisitorExecuteFunc.h
	VisitorScheduler.h
	VisualVisitor.h
	WriteStateVisitor.h
	XMLPrintVisitor.h
	xml/BaseElement.h
	xml/ElementNameHelper.h
	xml/Element.h
	xml/Element.inl
	xml/NodeElement.h
	xml/ObjectElement.h
	xml/BaseMultiMappingElement.h
	xml/AttributeElement.h
	xml/DataElement.h
	xml/XML.h   
	SceneLoaderXML.h
	SceneLoaderPHP.h
	SceneLoaderFactory.h
	)

set(SOURCE_FILES

	common.cpp
	Node.cpp
	AnimateBeginEvent.cpp
	AnimateEndEvent.cpp
	AnimateVisitor.cpp
	BehaviorUpdatePositionVisitor.cpp
	CactusStackStorage.cpp
	CleanupVisitor.cpp
	CollisionAnimationLoop.cpp
	CollisionVisitor.cpp
	CopyAspectVisitor.cpp
	DeactivatedNodeVisitor.cpp
	DefaultAnimationLoop.cpp
	DefaultVisualManagerLoop.cpp
	DeleteVisitor.cpp
	ExportGnuplotVisitor.cpp
	ExportOBJVisitor.cpp
	InitVisitor.cpp   
    MechanicalComputeEnergyVisitor.cpp
	MechanicalOperations.cpp
	MechanicalMatrixVisitor.cpp
	MechanicalVPrintVisitor.cpp
	MechanicalVisitor.cpp   
	MutationListener.cpp 
	ParallelVisitorScheduler.cpp
	PauseEvent.cpp
	PipelineImpl.cpp
	PositionEvent.cpp
	PrintVisitor.cpp
	PropagateEventVisitor.cpp
	ReleaseAspectVisitor.cpp
	ResetVisitor.cpp
	Simulation.cpp  
	SolveVisitor.cpp
	TopologyChangeVisitor.cpp
	StateChangeVisitor.cpp   
	TransformationVisitor.cpp
	UpdateBoundingBoxVisitor.cpp
	UpdateContextVisitor.cpp
	UpdateLinksVisitor.cpp
	UpdateMappingEndEvent.cpp
	UpdateMappingVisitor.cpp
	VectorOperations.cpp
	VelocityThresholdVisitor.cpp
	Visitor.cpp
	VisitorScheduler.cpp
	VisualVisitor.cpp
	WriteStateVisitor.cpp
	XMLPrintVisitor.cpp
	xml/BaseElement.cpp
	xml/ElementNameHelper.cpp
	xml/NodeElement.cpp
	xml/ObjectElement.cpp
	xml/BaseMultiMappingElement.cpp
	xml/AttributeElement.cpp
	xml/DataElement.cpp
	xml/XML.cpp 
	SceneLoaderFactory.cpp
	SceneLoaderXML.cpp
	SceneLoaderPHP.cpp
	)

if(SOFA-MISC_SMP)
	list(APPEND HEADER_FILES "ChangeListener.h")
	list(APPEND HEADER_FILES "ParallelMechanicalVisitor.h")
	list(APPEND SOURCE_FILES "ChangeListener.cpp")
	list(APPEND SOURCE_FILES "ParallelMechanicalVisitor.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SIMULATION_COMMON")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaCore tinyxml)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

