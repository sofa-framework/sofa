cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Controller LANGUAGES CXX)

set(SOFACOMPONENTCONTROLLER_SOURCE_DIR "src/sofa/component/controller")

set(HEADER_FILES
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/init.h
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/Controller.h
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/MechanicalStateController.h
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/MechanicalStateController.inl
)

set(SOURCE_FILES
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/init.cpp
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/Controller.cpp
    ${SOFACOMPONENTCONTROLLER_SOURCE_DIR}/MechanicalStateController.cpp

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
