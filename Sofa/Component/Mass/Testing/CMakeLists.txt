cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Mass.Testing)

set(SOFACOMPONENTMAPPINGTESTING_SRC "src/sofa/component/mass/testing")

set(HEADER_FILES
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/MassTestCreation.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/empty.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)

target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
