cmake_minimum_required(VERSION 3.1)
project(SofaSimulationCore)

set(SRC_ROOT "src/sofa/simulation")

set(HEADER_FILES
    ${SRC_ROOT}/AnimateBeginEvent.h
    ${SRC_ROOT}/AnimateEndEvent.h
    ${SRC_ROOT}/AnimateVisitor.h
    ${SRC_ROOT}/BehaviorUpdatePositionVisitor.h
    ${SRC_ROOT}/CactusStackStorage.h
    ${SRC_ROOT}/ClassSystem.h
    ${SRC_ROOT}/CleanupVisitor.h
    ${SRC_ROOT}/CollisionAnimationLoop.h
    ${SRC_ROOT}/CollisionBeginEvent.h
    ${SRC_ROOT}/CollisionEndEvent.h
    ${SRC_ROOT}/CollisionVisitor.h
    ${SRC_ROOT}/Colors.h
    ${SRC_ROOT}/DeactivatedNodeVisitor.h
    ${SRC_ROOT}/DefaultAnimationLoop.h
    ${SRC_ROOT}/DefaultVisualManagerLoop.h
    ${SRC_ROOT}/DeleteVisitor.h
    ${SRC_ROOT}/ExportGnuplotVisitor.h
    ${SRC_ROOT}/ExportOBJVisitor.h
    ${SRC_ROOT}/InitVisitor.h
    ${SRC_ROOT}/IntegrateBeginEvent.h
    ${SRC_ROOT}/IntegrateEndEvent.h
    ${SRC_ROOT}/LocalStorage.h
    ${SRC_ROOT}/MechanicalComputeEnergyVisitor.h
    ${SRC_ROOT}/MechanicalMatrixVisitor.h
    ${SRC_ROOT}/MechanicalOperations.h
    ${SRC_ROOT}/MechanicalVPrintVisitor.h
    ${SRC_ROOT}/MechanicalVisitor.h
    ${SRC_ROOT}/MutationListener.h
    ${SRC_ROOT}/Node.h
    ${SRC_ROOT}/Node.inl
    ${SRC_ROOT}/ParallelVisitorScheduler.h
    ${SRC_ROOT}/PauseEvent.h
    ${SRC_ROOT}/PipelineImpl.h
    ${SRC_ROOT}/PositionEvent.h
    ${SRC_ROOT}/PrintVisitor.h
    ${SRC_ROOT}/PropagateEventVisitor.h
    ${SRC_ROOT}/ResetVisitor.h
    ${SRC_ROOT}/SceneLoaderFactory.h
    ${SRC_ROOT}/Simulation.h
    ${SRC_ROOT}/SolveVisitor.h
    ${SRC_ROOT}/StateChangeVisitor.h
    ${SRC_ROOT}/TopologyChangeVisitor.h
    ${SRC_ROOT}/UpdateBoundingBoxVisitor.h
    ${SRC_ROOT}/UpdateContextVisitor.h
    ${SRC_ROOT}/UpdateInternalDataVisitor.h
    ${SRC_ROOT}/UpdateLinksVisitor.h
    ${SRC_ROOT}/UpdateMappingEndEvent.h
    ${SRC_ROOT}/UpdateMappingVisitor.h
    ${SRC_ROOT}/VectorOperations.h
    ${SRC_ROOT}/VelocityThresholdVisitor.h
    ${SRC_ROOT}/Visitor.h
    ${SRC_ROOT}/VisitorExecuteFunc.h
    ${SRC_ROOT}/VisitorScheduler.h
    ${SRC_ROOT}/VisualVisitor.h
    ${SRC_ROOT}/WriteStateVisitor.h
    ${SRC_ROOT}/XMLPrintVisitor.h
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/MechanicalGetMomentumVisitor.h
    ${SRC_ROOT}/BaseSimulationExporter.h
    ${SRC_ROOT}/TaskScheduler.h
    ${SRC_ROOT}/DefaultTaskScheduler.h
    ${SRC_ROOT}/Task.h
    ${SRC_ROOT}/InitTasks.h
    ${SRC_ROOT}/Locks.h
    ${SRC_ROOT}/VisitorAsync.h
    ${SRC_ROOT}/events/SimulationInitDoneEvent.h
    ${SRC_ROOT}/events/SimulationInitStartEvent.h
    ${SRC_ROOT}/events/SimulationStartEvent.h
    ${SRC_ROOT}/events/SimulationStopEvent.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/AnimateBeginEvent.cpp
    ${SRC_ROOT}/AnimateEndEvent.cpp
    ${SRC_ROOT}/AnimateVisitor.cpp
    ${SRC_ROOT}/BehaviorUpdatePositionVisitor.cpp
    ${SRC_ROOT}/CactusStackStorage.cpp
    ${SRC_ROOT}/CleanupVisitor.cpp
    ${SRC_ROOT}/CollisionAnimationLoop.cpp
    ${SRC_ROOT}/CollisionBeginEvent.cpp
    ${SRC_ROOT}/CollisionEndEvent.cpp
    ${SRC_ROOT}/CollisionVisitor.cpp
    ${SRC_ROOT}/DeactivatedNodeVisitor.cpp
    ${SRC_ROOT}/DefaultAnimationLoop.cpp
    ${SRC_ROOT}/DefaultVisualManagerLoop.cpp
    ${SRC_ROOT}/DeleteVisitor.cpp
    ${SRC_ROOT}/ExportGnuplotVisitor.cpp
    ${SRC_ROOT}/ExportOBJVisitor.cpp
    ${SRC_ROOT}/InitVisitor.cpp
    ${SRC_ROOT}/IntegrateBeginEvent.cpp
    ${SRC_ROOT}/IntegrateEndEvent.cpp
    ${SRC_ROOT}/MechanicalMatrixVisitor.cpp
    ${SRC_ROOT}/MechanicalOperations.cpp
    ${SRC_ROOT}/MechanicalVPrintVisitor.cpp
    ${SRC_ROOT}/MechanicalVisitor.cpp
    ${SRC_ROOT}/MutationListener.cpp
    ${SRC_ROOT}/Node.cpp
    ${SRC_ROOT}/ParallelVisitorScheduler.cpp
    ${SRC_ROOT}/PauseEvent.cpp
    ${SRC_ROOT}/PipelineImpl.cpp
    ${SRC_ROOT}/PositionEvent.cpp
    ${SRC_ROOT}/PrintVisitor.cpp
    ${SRC_ROOT}/PropagateEventVisitor.cpp
    ${SRC_ROOT}/ResetVisitor.cpp
    ${SRC_ROOT}/SceneLoaderFactory.cpp
    ${SRC_ROOT}/Simulation.cpp
    ${SRC_ROOT}/SolveVisitor.cpp
    ${SRC_ROOT}/StateChangeVisitor.cpp
    ${SRC_ROOT}/TopologyChangeVisitor.cpp
    ${SRC_ROOT}/UpdateBoundingBoxVisitor.cpp
    ${SRC_ROOT}/UpdateContextVisitor.cpp
    ${SRC_ROOT}/UpdateInternalDataVisitor.cpp
    ${SRC_ROOT}/UpdateLinksVisitor.cpp
    ${SRC_ROOT}/UpdateMappingEndEvent.cpp
    ${SRC_ROOT}/UpdateMappingVisitor.cpp
    ${SRC_ROOT}/VectorOperations.cpp
    ${SRC_ROOT}/VelocityThresholdVisitor.cpp
    ${SRC_ROOT}/Visitor.cpp
    ${SRC_ROOT}/VisitorScheduler.cpp
    ${SRC_ROOT}/VisualVisitor.cpp
    ${SRC_ROOT}/WriteStateVisitor.cpp
    ${SRC_ROOT}/XMLPrintVisitor.cpp
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/MechanicalComputeEnergyVisitor.cpp
    ${SRC_ROOT}/BaseSimulationExporter.cpp
    ${SRC_ROOT}/TaskScheduler.cpp
    ${SRC_ROOT}/DefaultTaskScheduler.cpp
    ${SRC_ROOT}/Task.cpp
    ${SRC_ROOT}/InitTasks.cpp
    ${SRC_ROOT}/events/SimulationInitDoneEvent.cpp
    ${SRC_ROOT}/events/SimulationInitStartEvent.cpp
    ${SRC_ROOT}/events/SimulationStartEvent.cpp
    ${SRC_ROOT}/events/SimulationStopEvent.cpp
)
if(SOFA_BUILD_TESTS)
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/testing/Node_test.h
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_CORE")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")

# is this a compiler/linker version specific problem?
if(CMAKE_SYSTEM_NAME STREQUAL Linux)
    # target_link_options is  available from CMake 3.13
    # target_link_options(${PROJECT_NAME} PUBLIC "-pthread")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread" )
endif()

sofa_install_targets(SofaFramework ${PROJECT_NAME} "SofaFramework/sofa/simulation" ${SRC_ROOT})

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaSimulationCore_test)
endif()
