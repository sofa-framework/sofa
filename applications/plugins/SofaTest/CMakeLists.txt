cmake_minimum_required(VERSION 3.12)
project(SofaTest VERSION 0.1)

set(HEADER_FILES
    InitPlugin_test.h
    Sofa_test.h
    )

set(SOURCE_FILES
    InitPlugin_test.cpp
    Sofa_test.cpp
    )

list(APPEND HEADER_FILES
    BroadPhase_test.h
    DataEngine_test.h
    ForceField_test.h
    PrimitiveCreation.h
    Elasticity_test.h
    Elasticity_test.inl
    )
list(APPEND SOURCE_FILES
    PrimitiveCreation.cpp
    Elasticity_test.cpp
    )

sofa_find_package(Sofa.Framework REQUIRED)
sofa_find_package(Sofa.Component REQUIRED)
sofa_find_package(SceneCreator REQUIRED)
sofa_find_package(Sofa.Testing REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SceneCreator Sofa.Framework Sofa.Component Sofa.Testing)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
