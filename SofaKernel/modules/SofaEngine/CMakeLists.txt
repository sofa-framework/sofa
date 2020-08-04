cmake_minimum_required(VERSION 3.12)
project(SofaEngine)

set(HEADER_FILES
    BoxROI.h
    BoxROI.inl
    config.h
    initEngine.h
)

set(SOURCE_FILES
    BoxROI.cpp
    initEngine.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaCore SofaDefaultType SofaSimulationCommon)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaCommon
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
