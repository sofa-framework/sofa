cmake_minimum_required(VERSION 3.1)
project(SofaValidation)

message(STATUS "SofaValidation:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initValidation.h
)

set(SOURCE_FILES
    initValidation.cpp
)

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
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
        )
    list(APPEND SOURCE_FILES
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
        )

    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()



add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseCollision SofaLoader SofaMeshCollision SofaGeneralLoader)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VALIDATION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})

## Add test project
if(SOFA_BUILD_TESTS)
  add_subdirectory(SofaValidation_test)
endif()
