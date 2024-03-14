cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Mapping.Testing)

set(SOFACOMPONENTMAPPINGTESTING_SRC "src/sofa/component/mapping/testing")

set(HEADER_FILES
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/MappingTestCreation.h
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/Multi2MappingTestCreation.h
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/MultiMappingTestCreation.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPINGTESTING_SRC}/empty.cpp 
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)

target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
