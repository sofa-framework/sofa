cmake_minimum_required(VERSION 3.12)
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

find_package(SofaEigen2Solver REQUIRED)
find_package(Compliant REQUIRED)
find_package(Flexible QUIET)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaEigen2Solver)
target_link_libraries(${PROJECT_NAME} Compliant)
set(PREASSEMBLEDMASS_COMPILER_FLAGS "-DSOFA_BUILD_PreassembledMass")

if(Flexible_FOUND)
    target_link_libraries(${PROJECT_NAME} Flexible)
    set(PREASSEMBLEDMASS_COMPILER_FLAGS "${PREASSEMBLEDMASS_COMPILER_FLAGS} -DSOFA_HAVE_FLEXIBLE")
else()
    message(STATUS "PreassembledMass: could not find Flexible")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS ${PREASSEMBLEDMASS_COMPILER_FLAGS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
