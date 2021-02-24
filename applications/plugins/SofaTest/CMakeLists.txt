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

set(PYTHON_FILES "")

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

find_package(SofaFramework REQUIRED)
find_package(SofaMiscForceField REQUIRED)
find_package(SofaMiscMapping REQUIRED)
find_package(SofaMiscFem REQUIRED)
find_package(SceneCreator REQUIRED)

sofa_find_package(SofaPython QUIET)
if(SofaPython_FOUND)
    list(APPEND HEADER_FILES "Python_test.h")
    list(APPEND SOURCE_FILES "Python_test.cpp")
    set(PYTHON_FILES
        python/SofaTest/Macro.py
        python/SofaTest/__init__.py
        )
    message(STATUS "SofaTest: optional dependency SofaPython found, Python_test will be built")
else()
    message(STATUS "SofaTest: optional dependency SofaPython NOT found, Python_test won't be built")
endif()

# Config files and install rules for pythons scripts
sofa_install_pythonscripts(PLUGIN_NAME ${PROJECT_NAME} PYTHONSCRIPTS_SOURCE_DIR "python")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${PYTHON_FILES})
if(SofaPython_FOUND)
    target_link_libraries(${PROJECT_NAME} SofaPython)
endif()
target_link_libraries(${PROJECT_NAME} SceneCreator SofaBase SofaCommon SofaGeneral SofaMiscForceField SofaMiscMapping SofaMiscFem)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    PACKAGE_VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
