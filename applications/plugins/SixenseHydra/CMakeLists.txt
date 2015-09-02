cmake_minimum_required(VERSION 2.8)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

project("SixenseHydra")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    RazerHydraDriver.h
    config.h
)

set(SOURCE_FILES
    RazerHydraDriver.cpp
    initSixenseHydra.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set(COMPILER_DEFINES "SOFA_BUILD_SIXENSE_HYDRA")
set(LINKER_DEPENDENCIES SofaBaseLinearSolver SofaConstraint SofaUserInteraction SofaSimpleFem SofaBaseTopology SofaGuiCommon SofaMeshCollision)

# add Sixense SDK
find_package(Sixense REQUIRED)
include_directories(${SIXENSE_INCLUDE_DIR})
TARGET_LINK_LIBRARIES(SixenseHydra ${SIXENSE_LIBRARY} ${SIXENSE_UTILS_LIBRARY})

include(${SOFA_CMAKE_DIR}/postProject.cmake)