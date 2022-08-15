cmake_minimum_required(VERSION 3.12)
project(SofaMiscFem LANGUAGES CXX)

set(SOFAMISCFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCFEM_SRC}/config.h.in
    ${SOFAMISCFEM_SRC}/initSofaMiscFem.h
    )

set(SOURCE_FILES
    ${SOFAMISCFEM_SRC}/initSofaMiscFem.cpp
    )
    
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.SolidMechanics.FEM.Elastic, Sofa.Component.SolidMechanics.FEM.HyperElastic and Sofa.Component.SolidMechanics.TensorMass instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.SolidMechanics.FEM.Elastic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.FEM.HyperElastic REQUIRED)
sofa_find_package(Sofa.Component.SolidMechanics.TensorMass REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.Elastic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.FEM.HyperElastic)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.SolidMechanics.TensorMass)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )

