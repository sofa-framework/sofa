cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.ODESolver LANGUAGES CXX)

set(SOFACOMPONENTODESOLVER_SOURCE_DIR "src/sofa/component/odesolver")

set(SOFACOMPONENTODESOLVER_DIRS Forward Backward)
set(SOFACOMPONENTODESOLVER_TARGETS)
foreach(odesolver_dir ${SOFACOMPONENTODESOLVER_DIRS})
    sofang_add_component_subdirectory(${odesolver_dir} ${PROJECT_NAME}.${odesolver_dir})
    list(APPEND SOFACOMPONENTODESOLVER_TARGETS ${PROJECT_NAME}.${odesolver_dir})
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

cmake_dependent_option(SOFANG_ENABLE_TESTING_SOFA_COMPONENT_ODESOLVER "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFANG_ENABLE_TESTING_SOFA_COMPONENT_ODESOLVER)
    # add testing tools (test templates, etc)
    add_subdirectory(Testing)
    # add tests
    add_subdirectory(tests)
endif()
