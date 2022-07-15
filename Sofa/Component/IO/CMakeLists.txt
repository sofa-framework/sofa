cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.IO LANGUAGES CXX)

set(SOFACOMPONENTIO_SOURCE_DIR "src/sofa/component/io")

set(SOFACOMPONENTIO_DIRS Mesh)
set(SOFACOMPONENTIO_TARGETS)
foreach(io_dir ${SOFACOMPONENTIO_DIRS})
    sofa_add_subdirectory(module ${io_dir} ${PROJECT_NAME}.${io_dir} ON)
    if(TARGET ${PROJECT_NAME}.${io_dir})
        list(APPEND SOFACOMPONENTIO_TARGETS ${PROJECT_NAME}.${io_dir})
    endif()
endforeach()


set(HEADER_FILES
    ${SOFACOMPONENTIO_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTIO_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTIO_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTIO_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
