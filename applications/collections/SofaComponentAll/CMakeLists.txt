cmake_minimum_required(VERSION 3.12)
project(SofaComponentAll VERSION 1.0)

find_package(SofaFramework REQUIRED)
find_package(SofaBase REQUIRED)

# link with all modules
get_property(_allTargets GLOBAL PROPERTY __GlobalTargetList__)
get_property(_allTargetNames GLOBAL PROPERTY __GlobalTargetNameList__)

list(LENGTH _allTargets nbTargets)
math(EXPR len "${nbTargets} - 1")

set(_modulePrefix "MODULE")
set(MODULE_LIST "")
foreach(counter RANGE ${len})
    list(GET _allTargets ${counter} _target)
    list(GET _allTargetNames ${counter} _targetName)
    string(SUBSTRING "${_targetName}" 0 6 _testPrefix)
    if(${_testPrefix} MATCHES "${_modulePrefix}.*")
        find_package(${_target} REQUIRED)
        list(APPEND MODULE_LIST ${_target})
    endif()
endforeach()

set(SOFACOMPONENTALL_SRC_DIR src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFACOMPONENTALL_SRC_DIR}/config.h.in
    ${SOFACOMPONENTALL_SRC_DIR}/initComponentAll.h
    )
set(SOURCE_FILES
    ${SOFACOMPONENTALL_SRC_DIR}/initComponentAll.cpp
    )
set(EXTRA_FILES
    README.md
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${EXTRA_FILES})
target_link_libraries(${PROJECT_NAME} SofaBase ${MODULE_LIST})

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR "SofaComponentAll"
    INCLUDE_SOURCE_DIR "src"
    )
