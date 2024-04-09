cmake_minimum_required(VERSION 3.22)
project(SofaBase)

set(SOFABASE_MODULES
    SofaBaseLinearSolver
    SofaEigen2Solver
    SofaBaseTopology
    SofaBaseCollision
    SofaBaseMechanics
    SofaBaseVisual
    SofaBaseUtils
    )

foreach(TARGET ${SOFABASE_MODULES})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
    set_target_properties(${TARGET} PROPERTIES FOLDER SofaBase)
endforeach()

set(SOFABASE_SRC src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFABASE_SRC}/config.h.in
    ${SOFABASE_SRC}/initSofaBase.h
    )
set(SOURCE_FILES
    ${SOFABASE_SRC}/initSofaBase.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFABASE_MODULES})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER SofaBase)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
#    RELOCATABLE "collections" # disabled because we need SofaBase to be directly available
    )
