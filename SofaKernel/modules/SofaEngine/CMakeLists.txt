cmake_minimum_required(VERSION 3.1)
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

sofa_install_targets(SofaCommon ${PROJECT_NAME} "SofaCommon/${PROJECT_NAME}")
