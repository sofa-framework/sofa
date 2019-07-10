cmake_minimum_required(VERSION 3.1)
project(PreassembledMass VERSION 0.1)


set(HEADER_FILES
    PreassembledMass.h
    PreassembledMass.inl
    config.h
)

set(SOURCE_FILES
    PreassembledMass.cpp
    initPlugin.cpp
)

find_package(SofaGeneral REQUIRED)
find_package(Compliant REQUIRED)
find_package(Flexible QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEigen2Solver)
target_link_libraries(${PROJECT_NAME} Compliant)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")
set(PREASSEMBLEDMASS_COMPILER_FLAGS "-DSOFA_BUILD_PreassembledMass")

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
    set(PREASSEMBLEDMASS_COMPILER_FLAGS "${PREASSEMBLEDMASS_COMPILER_FLAGS} -DSOFA_HAVE_FLEXIBLE")
else()
    message(STATUS "PreassembledMass: could not find Flexible")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${PREASSEMBLEDMASS_COMPILER_FLAGS})

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
