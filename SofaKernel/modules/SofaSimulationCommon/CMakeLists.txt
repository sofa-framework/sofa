cmake_minimum_required(VERSION 3.12)
project(SofaSimulationCommon LANGUAGES CXX)

set(SOFASIMULATIONCOMMON_SRC "src/${PROJECT_NAME}")

sofa_find_package(TinyXML REQUIRED)

set(HEADER_FILES
    ${SOFASIMULATIONCOMMON_SRC}/config.h.in
    ${SOFASIMULATIONCOMMON_SRC}/common.h
    ${SOFASIMULATIONCOMMON_SRC}/init.h
    ${SOFASIMULATIONCOMMON_SRC}/FindByTypeVisitor.h
    ${SOFASIMULATIONCOMMON_SRC}/SceneLoaderPHP.h
    ${SOFASIMULATIONCOMMON_SRC}/SceneLoaderXML.h
    ${SOFASIMULATIONCOMMON_SRC}/TransformationVisitor.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/AttributeElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/BaseElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/BaseMultiMappingElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/DataElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/ElementNameHelper.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/Element.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/Element.inl
    ${SOFASIMULATIONCOMMON_SRC}/xml/NodeElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/ObjectElement.h
    ${SOFASIMULATIONCOMMON_SRC}/xml/XML.h
)

set(SOURCE_FILES
    ${SOFASIMULATIONCOMMON_SRC}/SceneLoaderPHP.cpp
    ${SOFASIMULATIONCOMMON_SRC}/SceneLoaderXML.cpp
    ${SOFASIMULATIONCOMMON_SRC}/TransformationVisitor.cpp
    ${SOFASIMULATIONCOMMON_SRC}/init.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/AttributeElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/BaseElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/BaseMultiMappingElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/DataElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/ElementNameHelper.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/NodeElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/ObjectElement.cpp
    ${SOFASIMULATIONCOMMON_SRC}/xml/XML.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaCore SofaSimulationCore)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml) # Private because not exported in API

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFASIMULATIONCOMMON_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFASIMULATIONCOMMON_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${PROJECT_NAME}_test)
endif()
