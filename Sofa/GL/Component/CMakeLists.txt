cmake_minimum_required(VERSION 3.12)
project(Sofa.GL.Component LANGUAGES CXX)

set(SOFAGLCOMPONENT_SOURCE_DIR "src/sofa/gl/component")

set(SOFAGLCOMPONENT_DIRS Rendering2D Rendering3D Engine Shader)
set(SOFAGLCOMPONENT_TARGETS)
foreach(gl_dir ${SOFAGLCOMPONENT_DIRS})
    sofa_add_subdirectory(module ${gl_dir} ${PROJECT_NAME}.${gl_dir} ON)
    if(TARGET ${PROJECT_NAME}.${gl_dir})
        list(APPEND SOFAGLCOMPONENT_TARGETS ${PROJECT_NAME}.${gl_dir})
    endif()
endforeach()


set(HEADER_FILES
    ${SOFAGLCOMPONENT_SOURCE_DIR}/config.h.in
    ${SOFAGLCOMPONENT_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFAGLCOMPONENT_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAGLCOMPONENT_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
