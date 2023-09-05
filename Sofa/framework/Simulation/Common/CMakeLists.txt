cmake_minimum_required(VERSION 3.12)
project(Sofa.Simulation.Common LANGUAGES CXX)

set(SOFA_SIMULATION_COMMON_SRC "src/sofa/simulation/common")

set(HEADER_FILES
    ${SOFA_SIMULATION_COMMON_SRC}/config.h.in
    ${SOFA_SIMULATION_COMMON_SRC}/init.h
    ${SOFA_SIMULATION_COMMON_SRC}/initSofaSimulationCommon.h
    ${SOFA_SIMULATION_COMMON_SRC}/FindByTypeVisitor.h
    ${SOFA_SIMULATION_COMMON_SRC}/SceneLoaderPHP.h
    ${SOFA_SIMULATION_COMMON_SRC}/SceneLoaderXML.h
    ${SOFA_SIMULATION_COMMON_SRC}/TransformationVisitor.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/AttributeElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/BaseElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/BaseMultiMappingElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/DataElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/Element.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/Element.inl
    ${SOFA_SIMULATION_COMMON_SRC}/xml/NodeElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/ObjectElement.h
    ${SOFA_SIMULATION_COMMON_SRC}/xml/XML.h
)

set(SOURCE_FILES
    ${SOFA_SIMULATION_COMMON_SRC}/SceneLoaderPHP.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/SceneLoaderXML.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/TransformationVisitor.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/init.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/initSofaSimulationCommon.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/AttributeElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/BaseElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/BaseMultiMappingElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/DataElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/NodeElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/ObjectElement.cpp
    ${SOFA_SIMULATION_COMMON_SRC}/xml/XML.cpp
)

sofa_find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(TinyXML REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Core Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml) # Private because not exported in API

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_SIMULATION_COMMON_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_SIMULATION_COMMON_BUILD_TESTS)
    add_subdirectory(test)
endif()
