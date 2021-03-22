cmake_minimum_required(VERSION 3.12)
project(Sofa.Compat LANGUAGES CXX)

set(SOFACOMPATSRC_ROOT "src/sofa")

set(HEADER_FILES
    # ${SOFACOMPATSRC_ROOT}/config.h.in
    )

set(SOURCE_FILES
    ${SOFACOMPATSRC_ROOT}/initSofaCompat.cpp
    )

sofa_find_package(Sofa.Config REQUIRED)
sofa_find_package(SofaFramework REQUIRED)
# sofa_find_package(Sofa.Type REQUIRED)
# sofa_find_package(Sofa.Geometry REQUIRED)
# sofa_find_package(Sofa.Topology REQUIRED)
# sofa_find_package(SofaHelper REQUIRED)
# sofa_find_package(SofaDefaulttype REQUIRED)
# sofa_find_package(SofaCore REQUIRED)
# sofa_find_package(SofaSimulationCoe REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Type Sofa.Geometry Sofa.Topology)
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDefaulttype SofaCore SofaSimulationCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
