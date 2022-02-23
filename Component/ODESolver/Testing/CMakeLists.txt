cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.ODESolver.Testing)

set(SOFACOMPONENTODESOLVERTESTING_SRC "src/sofa/component/odesolver/testing")

# only header files
set(HEADER_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/MassSpringSystemCreation.h
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/SunPlanetSystemCreation.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/empty.cpp # because of not interface type
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)
target_link_libraries(${PROJECT_NAME} Sofa.Config)
