cmake_minimum_required(VERSION 3.12)
project(SofaSimulationCommon)

set(SRC_DIR "src/SofaSimulationCommon")

set(HEADER_FILES
    ${SRC_DIR}/config.h.in
    ${SRC_DIR}/init.h
    ${SRC_DIR}/initSofaSimulationCommon.h
)

set(SOURCE_FILES
    ${SRC_DIR}/init.cpp
    ${SRC_DIR}/initSofaSimulationCommon.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Simulation.Common instead.")

sofa_find_package(Sofa.Simulation.Common REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Common)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
