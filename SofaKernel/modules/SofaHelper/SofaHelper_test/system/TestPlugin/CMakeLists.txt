cmake_minimum_required(VERSION 3.12)
project(TestPlugin VERSION 0.7)


set(HEADER_FILES
    ComponentA.h
    ComponentB.h
    TestPlugin.h
)

set(SOURCE_FILES
    ComponentA.cpp
    ComponentB.cpp
    initTestPlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} SofaCore SofaDefaultType)

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME}
    )
