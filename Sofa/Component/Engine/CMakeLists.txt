cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Engine LANGUAGES CXX)

set(SOFACOMPONENTENGINE_SOURCE_DIR "src/sofa/component/engine")

set(SOFACOMPONENTENGINE_DIRS Analyze Generate Select Transform)
set(SOFACOMPONENTENGINE_TARGETS)
foreach(engine_dir ${SOFACOMPONENTENGINE_DIRS})
    sofa_add_subdirectory(module ${engine_dir} ${PROJECT_NAME}.${engine_dir} ON)
    if(TARGET ${PROJECT_NAME}.${engine_dir})
        list(APPEND SOFACOMPONENTENGINE_TARGETS ${PROJECT_NAME}.${engine_dir})
    endif()
endforeach()


set(HEADER_FILES
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTENGINE_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTENGINE_TARGETS})

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
