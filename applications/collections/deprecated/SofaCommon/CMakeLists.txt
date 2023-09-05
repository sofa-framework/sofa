cmake_minimum_required(VERSION 3.12)
project(SofaCommon)

# Modules
set(SOFACOMMON_MODULES
    SofaSimpleFem
    SofaRigid
    SofaDeformable
    SofaObjectInteraction
    SofaEngine
    SofaExplicitOdeSolver
    SofaImplicitOdeSolver
    SofaLoader
)

foreach(module ${SOFACOMMON_MODULES})
    sofa_find_package(${module} QUIET)
    if(NOT ${module}_FOUND)
        string(TOUPPER module_${module} MODULE_TO_CHECK)
        list(APPEND LIST_MODULES_TO_CHECK ${MODULE_TO_CHECK})
    endif()
endforeach()
if(LIST_MODULES_TO_CHECK)
    message(FATAL_ERROR
        " If you still want to use ${PROJECT_NAME}, please enable the following CMake option(s): \n"
        " ${LIST_MODULES_TO_CHECK}"
        )
endif()

set(SOFACOMMON_SRC src/${PROJECT_NAME})
set(HEADER_FILES
    ${SOFACOMMON_SRC}/config.h.in
    ${SOFACOMMON_SRC}/initSofaCommon.h
    )
set(SOURCE_FILES
    ${SOFACOMMON_SRC}/initSofaCommon.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMMON_MODULES})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
    )
