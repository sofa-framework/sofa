cmake_minimum_required(VERSION 2.8.12)
project(SofaValidation)

set(HEADER_FILES

    config.h
    initValidation.h
    CompareState.h
    CompareTopology.h
    DevAngleCollisionMonitor.h
    DevAngleCollisionMonitor.inl
    DevTensionMonitor.h
    DevTensionMonitor.inl
    DevMonitorManager.h
    DevMonitor.h
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
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaBaseCollision SofaLoader SofaMeshCollision )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/SofaGeneral>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAVALIDATION_COMPILER_FLAGS "-DSOFA_BUILD_VALIDATION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAVALIDATION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION bin COMPONENT SofaGeneral_libraries
        LIBRARY DESTINATION lib COMPONENT SofaGeneral_libraries
        ARCHIVE DESTINATION lib COMPONENT SofaGeneral_libraries
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}" COMPONENT SofaGeneral_headers
)

