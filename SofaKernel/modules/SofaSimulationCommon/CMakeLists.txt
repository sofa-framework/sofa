cmake_minimum_required(VERSION 3.1)
project(SofaSimulationCommon)

set(HEADER_FILES
    FindByTypeVisitor.h
    SceneLoaderPHP.h
    SceneLoaderXML.h
    TransformationVisitor.h
    xml/AttributeElement.h
    xml/BaseElement.h
    xml/BaseMultiMappingElement.h
    xml/DataElement.h
    xml/ElementNameHelper.h
    xml/NodeElement.h
    xml/ObjectElement.h
    xml/XML.h
)

set(SOURCE_FILES
    SceneLoaderPHP.cpp
    SceneLoaderXML.cpp
    TransformationVisitor.cpp
    init.cpp
    xml/AttributeElement.cpp
    xml/BaseElement.cpp
    xml/BaseMultiMappingElement.cpp
    xml/DataElement.cpp
    xml/ElementNameHelper.cpp
    xml/NodeElement.cpp
    xml/ObjectElement.cpp
    xml/XML.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore)
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SIMULATION_COMMON")

sofa_install_targets(SofaSimulation ${PROJECT_NAME} ${PROJECT_NAME})
