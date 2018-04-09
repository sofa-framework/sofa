cmake_minimum_required(VERSION 3.1)
project(SofaValidation)

set(HEADER_FILES
    CompareState.h
    CompareTopology.h
    DataController.h
    DataMonitor.h
    DevAngleCollisionMonitor.h
    DevAngleCollisionMonitor.inl
    DevMonitor.h
    DevMonitorManager.h
    DevTensionMonitor.h
    DevTensionMonitor.inl
    EvalPointsDistance.h
    EvalPointsDistance.inl
    EvalSurfaceDistance.h
    EvalSurfaceDistance.inl
    ExtraMonitor.h
    ExtraMonitor.inl
    Monitor.h
    Monitor.inl
    config.h
    initValidation.h
    )

set(SOURCE_FILES
    CompareState.cpp
    CompareTopology.cpp
    DataController.cpp
    DataMonitor.cpp
    DevAngleCollisionMonitor.cpp
    DevMonitorManager.cpp
    DevTensionMonitor.cpp
    EvalPointsDistance.cpp
    EvalSurfaceDistance.cpp
    ExtraMonitor.cpp
    Monitor.cpp
    initValidation.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseCollision SofaLoader SofaMeshCollision SofaGeneralLoader)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VALIDATION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})

## Add test project
if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaValidation_test)
endif()
