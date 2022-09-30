cmake_minimum_required(VERSION 3.12)
project(SofaGeneralSimpleFem LANGUAGES CXX)

set(SOFAGENERALSIMPLEFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALSIMPLEFEM_SRC}/config.h.in
    ${SOFAGENERALSIMPLEFEM_SRC}/initSofaGeneralSimpleFem.h
    )

set(SOURCE_FILES
    ${SOFAGENERALSIMPLEFEM_SRC}/initSofaGeneralSimpleFem.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.SolidMechanics.FEM.Elastic instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Elastic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
