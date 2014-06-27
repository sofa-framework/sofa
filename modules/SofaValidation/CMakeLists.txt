cmake_minimum_required(VERSION 2.8)
project(SofaValidation)

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
target_link_libraries( ${PROJECT_NAME} SofaBaseCollision SofaLoader SofaMeshCollision )

set( SOFAVALIDATION_COMPILER_FLAGS "-DSOFA_BUILD_VALIDATION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAVALIDATION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaValidation_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)

