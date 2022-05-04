cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.StateContainer LANGUAGES CXX)

set(SOFACOMPONENTSTATECONTAINER_SOURCE_DIR "src/sofa/component/statecontainer")

set(HEADER_FILES
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/init.h
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MechanicalObject.h
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MechanicalObject.inl
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MappedObject.h
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MappedObject.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MappedObject.cpp
    ${SOFACOMPONENTSTATECONTAINER_SOURCE_DIR}/MechanicalObject.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_STATECONTAINER_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_STATECONTAINER_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
