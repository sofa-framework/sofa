cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Engine.Testing)

set(SOFACOMPONENTENGINETESTING_SRC "src/sofa/component/engine/testing")

set(HEADER_FILES
    ${SOFACOMPONENTENGINETESTING_SRC}/DataEngineTestCreation.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTENGINETESTING_SRC}/empty.cpp 
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)

target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
