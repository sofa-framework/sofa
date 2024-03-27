cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.SolidMechanics LANGUAGES CXX)

set(SOFACOMPONENTSOLIDMECHANICS_SOURCE_DIR "src/sofa/component/solidmechanics")

sofa_add_subdirectory_modules(SOFACOMPONENTSOLIDMECHANICS_TARGETS
    DIRECTORIES Spring FEM TensorMass
)

set(HEADER_FILES
    ${SOFACOMPONENTSOLIDMECHANICS_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTSOLIDMECHANICS_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTSOLIDMECHANICS_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTSOLIDMECHANICS_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

if(SOFA_BUILD_TESTS)
    add_subdirectory(Testing)
    add_subdirectory(simutests)
endif()
