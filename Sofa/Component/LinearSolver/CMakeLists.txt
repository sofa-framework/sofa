cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.LinearSolver LANGUAGES CXX)

set(SOFACOMPONENTLINEARSOLVER_SOURCE_DIR "src/sofa/component/linearsolver")

set(SOFACOMPONENTLINEARSOLVER_DIRS Iterative Direct Preconditioner)
set(SOFACOMPONENTLINEARSOLVER_TARGETS)
foreach(linearsolver_dir ${SOFACOMPONENTLINEARSOLVER_DIRS})
    sofa_add_subdirectory(module ${linearsolver_dir} ${PROJECT_NAME}.${linearsolver_dir} ON)
    if(TARGET ${PROJECT_NAME}.${linearsolver_dir})
        list(APPEND SOFACOMPONENTLINEARSOLVER_TARGETS ${PROJECT_NAME}.${linearsolver_dir})
    endif()
endforeach()

set(HEADER_FILES
    ${SOFACOMPONENTLINEARSOLVER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTLINEARSOLVER_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTLINEARSOLVER_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTLINEARSOLVER_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
