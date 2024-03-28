cmake_minimum_required(VERSION 3.22)

project(Sofa.Component.Collision.Testing)

set(SOFACOMPONENTCOLLISIONTESTING_SRC "src/sofa/component/collision/testing")

set(HEADER_FILES
    ${SOFACOMPONENTCOLLISIONTESTING_SRC}/MeshPrimitiveCreator.h
    ${SOFACOMPONENTCOLLISIONTESTING_SRC}/SpherePrimitiveCreator.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTCOLLISIONTESTING_SRC}/empty.cpp 
)
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC src/)

target_link_libraries(${PROJECT_NAME} Sofa.Config Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} Sofa.Component.Topology.Container.Constant)
target_link_libraries(${PROJECT_NAME} Sofa.Component.StateContainer)
