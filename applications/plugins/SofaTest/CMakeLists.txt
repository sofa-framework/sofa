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
    Elasticity_test.h
    Elasticity_test.inl
    ForceField_test.h
    Mapping_test.h
    MultiMapping_test.h
    Multi2Mapping_test.h
    PrimitiveCreation.h
    TestMessageHandler.h
    #LogMessage.h
    )
list(APPEND SOURCE_FILES
    Elasticity_test.cpp
    PrimitiveCreation.cpp
    LinearElasticity_test.cpp
    AffinePatch_test.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    list(APPEND HEADER_FILES
        Elasticity_test_deprecated.h
        Elasticity_test_deprecated.inl
        )
    list(APPEND SOURCE_FILES
        Elasticity_test_deprecated.cpp
        )
endif()

sofa_find_package(SofaFramework REQUIRED)
sofa_find_package(SofaMiscForceField REQUIRED)
sofa_find_package(SofaMiscMapping REQUIRED)
sofa_find_package(SofaMiscFem REQUIRED)
sofa_find_package(SofaComponentAll REQUIRED)
sofa_find_package(SceneCreator REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
target_link_libraries(${PROJECT_NAME} SceneCreator SofaComponentAll SofaMiscForceField SofaMiscMapping SofaMiscFem Sofa.Testing)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    PACKAGE_VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
