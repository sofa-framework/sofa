include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initValidation.h
    CompareState.h
    CompareTopology.h
    DevAngleCollisionMonitor.h
    DevAngleCollisionMonitor.inl
    DevTensionMonitor.h
    DevTensionMonitor.inl
    DevMonitorManager.h
    ExtraMonitor.h
    ExtraMonitor.inl
    Monitor.h
    Monitor.inl
    EvalPointsDistance.h
    EvalPointsDistance.inl
    EvalSurfaceDistance.h
    EvalSurfaceDistance.inl
    DataMonitor.h
    DataController.h

    )

set(SOURCE_FILES

    initValidation.cpp
    CompareState.cpp
    CompareTopology.cpp
    DevAngleCollisionMonitor.cpp
    DevTensionMonitor.cpp
    DevMonitorManager.cpp
    ExtraMonitor.cpp
    Monitor.cpp
    EvalPointsDistance.cpp
    EvalSurfaceDistance.cpp
    DataMonitor.cpp
    DataController.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_VALIDATION" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaBaseCollision SofaLoader SofaMeshCollision )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
