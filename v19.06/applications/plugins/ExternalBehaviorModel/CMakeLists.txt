cmake_minimum_required(VERSION 3.1)
project(ExternalBehaviorModel VERSION 0.1)


set(HEADER_FILES
    FEMGridBehaviorModel.h
    FEMGridBehaviorModel.inl
    config.h
)

set(SOURCE_FILES
    FEMGridBehaviorModel.cpp
    initExternalBehaviorModel.cpp
)


find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ExternalBehaviorModel")
target_link_libraries(${PROJECT_NAME} SofaComponentGeneral)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
