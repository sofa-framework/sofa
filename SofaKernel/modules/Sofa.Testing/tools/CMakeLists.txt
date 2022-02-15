cmake_minimum_required(VERSION 3.12)

project(Sofa.Testing.Tools)

set(SOFATESTINGTOOLS_SRC "src/sofa/testing/tools")

# only header files
set(HEADER_FILES
    ${SOFATESTINGTOOLS_SRC}/DataEngineTestCreation.h
    ${SOFATESTINGTOOLS_SRC}/MassSpringSystemCreation.h
    ${SOFATESTINGTOOLS_SRC}/SunPlanetSystemCreation.h
)

set(SOURCE_FILES
    ${SOFATESTINGTOOLS_SRC}/empty.cpp # because of not interface type
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)
target_link_libraries(${PROJECT_NAME} Sofa.Config)
