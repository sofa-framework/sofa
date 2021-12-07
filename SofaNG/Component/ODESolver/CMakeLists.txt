cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.ODESolver LANGUAGES CXX)

set(SOFACOMPONENTODESOLVER_SOURCE_DIR "src/sofa/component/odesolver")

set(SOFACOMPONENTODESOLVER_DIRS Forward Backward)
set(SOFACOMPONENTODESOLVER_TARGETS)
foreach(odesolver_dir ${SOFACOMPONENTODESOLVER_DIRS})
    sofang_add_component_subdirectory(${odesolver_dir})
    list(APPEND SOFACOMPONENTODESOLVER_TARGETS ${PROJECT_NAME}.${odesolver_dir})
endforeach()

set(HEADER_FILES
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/init.cpp
)

# configure_file(${SOFACOMPONENTODESOLVER_SOURCE_DIR}/config.h.in 
#     ${CMAKE_BINARY_DIR}/include/sofa/component/odesolver/config.h)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
# set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTODESOLVER_TARGETS})

# target_compile_definitions(${PROJECT_NAME} PRIVATE BUILD_SOFACOMPONENTODESOLVER)
# target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
# target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/")
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
