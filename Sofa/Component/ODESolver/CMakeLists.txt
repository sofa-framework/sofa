cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.ODESolver LANGUAGES CXX)

set(SOFACOMPONENTODESOLVER_SOURCE_DIR "src/sofa/component/odesolver")

set(SOFACOMPONENTODESOLVER_DIRS Forward Backward)
set(SOFACOMPONENTODESOLVER_TARGETS)
foreach(odesolver_dir ${SOFACOMPONENTODESOLVER_DIRS})
    sofa_add_subdirectory(module ${odesolver_dir} ${PROJECT_NAME}.${odesolver_dir} ON)
    if(TARGET ${PROJECT_NAME}.${odesolver_dir})
        list(APPEND SOFACOMPONENTODESOLVER_TARGETS ${PROJECT_NAME}.${odesolver_dir})
    endif()
endforeach()

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTODESOLVER_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

if(SOFA_BUILD_TESTS)
    add_subdirectory(Testing)
endif()
