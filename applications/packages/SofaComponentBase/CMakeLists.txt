cmake_minimum_required(VERSION 3.1)
project(SofaComponentBase)

find_package(SofaBase REQUIRED)

set(HEADER_FILES
    config.h
    initComponentBase.h
    messageHandlerComponent.h
    MakeAliasComponent.h
    MakeDataAliasComponent.h
    InfoComponent.h
)

set(SOURCE_FILES
    initComponentBase.cpp
    messageHandlerComponent.cpp
    MakeAliasComponent.cpp
    MakeDataAliasComponent.cpp
    InfoComponent.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseMechanics SofaBaseCollision SofaBaseLinearSolver SofaBaseVisual SofaBaseUtils)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_BASE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")

sofa_create_package(${PROJECT_NAME} "1.0" ${PROJECT_NAME} ${PROJECT_NAME})
