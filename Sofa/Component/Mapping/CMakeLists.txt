cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Mapping LANGUAGES CXX)

set(SOFACOMPONENTMAPPING_SOURCE_DIR "src/sofa/component/mapping")

set(SOFACOMPONENTMAPPING_DIRS Linear NonLinear MappedMatrix)
set(SOFACOMPONENTMAPPING_TARGETS)
foreach(mapping_dir ${SOFACOMPONENTMAPPING_DIRS})
    sofa_add_subdirectory(module ${mapping_dir} ${PROJECT_NAME}.${mapping_dir} ON)
    if(TARGET ${PROJECT_NAME}.${mapping_dir})
        list(APPEND SOFACOMPONENTMAPPING_TARGETS ${PROJECT_NAME}.${mapping_dir})
    endif()
endforeach()


set(HEADER_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPING_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTMAPPING_TARGETS})

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
