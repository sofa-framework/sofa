cmake_minimum_required(VERSION 3.22)
project(SofaGeneralRigid LANGUAGES CXX)

set(SOFAGENERALRIGID_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALRIGID_SRC}/config.h.in
    ${SOFAGENERALRIGID_SRC}/initSofaGeneralRigid.h
    )

set(SOURCE_FILES
    ${SOFAGENERALRIGID_SRC}/initSofaGeneralRigid.cpp
    )


sofa_find_package(SofaBaseMechanics REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseMechanics)

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Mapping instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Mapping REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Mapping)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
