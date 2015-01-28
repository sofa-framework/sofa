include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initValidation.h 
     ../misc/CompareState.h
    # ../misc/CompareTopology.h 
    # ../misc/DevAngleCollisionMonitor.h 
    # ../misc/DevAngleCollisionMonitor.inl 
    # ../misc/DevTensionMonitor.h 
    # ../misc/DevTensionMonitor.inl 
    # ../misc/DevMonitorManager.h 
    # ../misc/ExtraMonitor.h 
    # ../misc/ExtraMonitor.inl 
    # ../misc/Monitor.h 
    # ../misc/Monitor.inl 
    # ../misc/EvalPointsDistance.h 
    # ../misc/EvalPointsDistance.inl 
    # ../misc/EvalSurfaceDistance.h 
    # ../misc/EvalSurfaceDistance.inl 
    # ../misc/DataMonitor.h 
    # ../misc/DataController.h 

    )
    
set(SOURCE_FILES

    ../initValidation.cpp 
     ../misc/CompareState.cpp
    # ../misc/CompareTopology.cpp 
    # ../misc/DevAngleCollisionMonitor.cpp 
    # ../misc/DevTensionMonitor.cpp 
    # ../misc/DevMonitorManager.cpp 
    # ../misc/ExtraMonitor.cpp 
    # ../misc/Monitor.cpp 
    # ../misc/EvalPointsDistance.cpp 
    # ../misc/EvalSurfaceDistance.cpp 
    # ../misc/DataMonitor.cpp 
    # ../misc/DataController.cpp 
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_VALIDATION" )
AddLinkerDependencies(${ZLIB_LIBRARIES_OPTIONAL} SofaBaseCollision SofaLoader SofaMeshCollision )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
