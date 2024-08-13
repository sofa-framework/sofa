cmake_minimum_required(VERSION 3.22)
project(SofaNewmat LANGUAGES CXX)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)

# Newmat
#sofa_find_package(Newmat QUIET) # useless IMO
if(NOT Newmat_FOUND)
    add_subdirectory(extlibs/newmat newmat)
endif()
set_target_properties(newmat PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(newmat PROPERTIES FOLDER "SofaExtlibs")



set(SOFANEWMAT_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFANEWMAT_SRC}/config.h.in
    ${SOFANEWMAT_SRC}/initSofaNewmat.h
    ${SOFANEWMAT_SRC}/LULinearSolver.h
    ${SOFANEWMAT_SRC}/NewMatMatrix.h
    ${SOFANEWMAT_SRC}/NewMatVector.h
)

set(SOURCE_FILES
    ${SOFANEWMAT_SRC}/initSofaNewmat.cpp
    ${SOFANEWMAT_SRC}/LULinearSolver.cpp
    ${SOFANEWMAT_SRC}/NewMatCGLinearSolver.cpp
    ${SOFANEWMAT_SRC}/NewMatCholeskySolver.cpp
)

add_library(${PROJECT_NAME} SHARED  ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC newmat Sofa.Component.LinearSolver.Direct)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)

cmake_dependent_option(SOFANEWMAT_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFANEWMAT_BUILD_TESTS)
    add_subdirectory(${PROJECT_NAME}_test)
endif()
