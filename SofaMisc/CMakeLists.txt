cmake_minimum_required(VERSION 3.12)
project(SofaMisc)

## Subdirectories
set(SOFAMISC_TARGETS
    SofaMiscExtra
    SofaMiscEngine
    SofaMiscFem
    SofaMiscForceField
    SofaMiscMapping
    SofaMiscSolver
    SofaMiscTopology
    )

foreach(TARGET ${SOFAMISC_TARGETS})
    add_subdirectory(../modules/${TARGET} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET})
endforeach()

set(SOFAMISC_SRC src/${PROJECT_NAME})

set(HEADER_FILES
    ${SOFAMISC_SRC}/config.h.in 
    ${SOFAMISC_SRC}/initSofaMisc.h
    )

set(SOURCE_FILES
    ${SOFAMISC_SRC}/initSofaMisc.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAMISC_TARGETS})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )
