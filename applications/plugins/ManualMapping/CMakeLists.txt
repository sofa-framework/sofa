cmake_minimum_required(VERSION 3.1)
project(ManualMapping VERSION 0.1)


set(HEADER_FILES
    ManualLinearMapping.h
    ManualLinearMapping.inl
    config.h
)

set(SOURCE_FILES
    ManualLinearMapping.cpp
    initManualMapping.cpp
)

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBase)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_${PROJECT_NAME}")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
