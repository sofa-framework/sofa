cmake_minimum_required(VERSION 3.12)

project(Sofa.Component.Topology.Testing)

set(SOFACOMPONENTODESOLVERTESTING_SRC "src/sofa/component/topology/testing")

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/fake_TopologyScene.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVERTESTING_SRC}/fake_TopologyScene.cpp 
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)
target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.SimulationCore)
