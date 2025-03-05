cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.SolidMechanics.Testing)

set(SOFACOMPONENTSOLIDMECHANICSTESTING_SRC "src/sofa/component/solidmechanics/testing")

set(HEADER_FILES
    ${SOFACOMPONENTSOLIDMECHANICSTESTING_SRC}/ForceFieldTestCreation.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTSOLIDMECHANICSTESTING_SRC}/empty.cpp 
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)

target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core SceneCreator)
