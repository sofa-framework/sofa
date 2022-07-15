cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Constraint LANGUAGES CXX)

set(SOFACOMPONENTCONSTRAINT_SOURCE_DIR "src/sofa/component/constraint")

set(SOFACOMPONENTCONSTRAINT_DIRS Lagrangian Projective)
set(SOFACOMPONENTCONSTRAINT_TARGETS)
foreach(constraint_dir ${SOFACOMPONENTCONSTRAINT_DIRS})
    sofa_add_subdirectory(module ${constraint_dir} ${PROJECT_NAME}.${constraint_dir} ON)
    if(TARGET ${PROJECT_NAME}.${constraint_dir})
        list(APPEND SOFACOMPONENTCONSTRAINT_TARGETS ${PROJECT_NAME}.${constraint_dir})
    endif()
endforeach()


set(HEADER_FILES
    ${SOFACOMPONENTCONSTRAINT_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTCONSTRAINT_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTCONSTRAINT_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTCONSTRAINT_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
