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
)

set(SOURCE_FILES
    ${SOFACOMPONENTODESOLVER_SOURCE_DIR}/init.cpp
)

set(TARGET_NAME ${PROJECT_NAME})

configure_file(${SOFACOMPONENTODESOLVER_SOURCE_DIR}/config.h.in 
    ${CMAKE_BINARY_DIR}/include/sofa/component/odesolver/config.h)

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${TARGET_NAME} PUBLIC ${SOFACOMPONENTODESOLVER_TARGETS})

target_compile_definitions(${TARGET_NAME} PRIVATE BUILD_SOFACOMPONENTODESOLVER)
target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/")
