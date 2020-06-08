cmake_minimum_required(VERSION 3.12)
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
target_link_libraries(${PROJECT_NAME} SofaGeneral)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_${PROJECT_NAME}")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
