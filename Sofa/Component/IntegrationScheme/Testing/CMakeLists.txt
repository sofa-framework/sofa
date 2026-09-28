cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.IntegrationScheme.Testing)

set(SOFACOMPONENTINTEGRATIONSCHEMETESTING_SRC "src/sofa/component/integrationscheme/testing")

set(HEADER_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMETESTING_SRC}/EigenTestUtilities.h
    ${SOFACOMPONENTINTEGRATIONSCHEMETESTING_SRC}/MassSpringSystemCreation.h
    ${SOFACOMPONENTINTEGRATIONSCHEMETESTING_SRC}/IntegrationSchemeSpringTest.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTINTEGRATIONSCHEMETESTING_SRC}/empty.cpp
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)
target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
