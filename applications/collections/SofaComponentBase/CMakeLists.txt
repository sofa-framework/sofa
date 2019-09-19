cmake_minimum_required(VERSION 3.1)
project(SofaComponentBase VERSION 1.0)

find_package(SofaBase REQUIRED)

set(HEADER_FILES
    config.h
    initComponentBase.h
    messageHandlerComponent.h
    MakeAliasComponent.h
    MakeDataAliasComponent.h
    InfoComponent.h
)

set(SOURCE_FILES
    initComponentBase.cpp
    messageHandlerComponent.cpp
    MakeAliasComponent.cpp
    MakeDataAliasComponent.cpp
    InfoComponent.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
foreach(target ${SOFABASE_TARGETS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${target})
endforeach()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_BASE")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/SofaKernel>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX _d)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}/${PROJECT_NAME}"
    )
