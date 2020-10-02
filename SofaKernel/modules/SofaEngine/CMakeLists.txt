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
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ENGINE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaCommon ${PROJECT_NAME} ${PROJECT_NAME})