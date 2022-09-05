cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Visual LANGUAGES CXX)

set(SOFACOMPONENTVISUAL_SOURCE_DIR "src/sofa/component/visual")

set(HEADER_FILES
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/init.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/BaseCamera.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/Camera.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/InteractiveCamera.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/LineAxis.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/RecordedCamera.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/Visual3DText.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualGrid.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualModelImpl.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualStyle.h
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualTransform.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/BaseCamera.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/Camera.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/InteractiveCamera.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/LineAxis.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/RecordedCamera.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/Visual3DText.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualGrid.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualModelImpl.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualStyle.cpp
    ${SOFACOMPONENTVISUAL_SOURCE_DIR}/VisualTransform.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(TinyXML REQUIRED) # BaseCamera
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED) # VisualModelImpl

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Grid)
target_link_libraries(${PROJECT_NAME} PRIVATE tinyxml)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_COMPONENT_VISUAL_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_COMPONENT_VISUAL_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
