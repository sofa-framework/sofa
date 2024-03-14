cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.ODESolver.Testing)

set(SOFACOMPONENTODESOLVERTESTING_SRC "src/sofa/component/odesolver/testing")

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/EigenTestUtilities.h
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/MassSpringSystemCreation.h
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/ODESolverSpringTest.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/empty.cpp 
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)
target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
