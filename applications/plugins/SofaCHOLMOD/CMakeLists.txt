cmake_minimum_required(VERSION 3.22)
project(SofaCHOLMOD LANGUAGES CXX)

find_package(Sofa.Config REQUIRED)

# CHOLMOD only supports double precision, so this plugin requires SOFA to be
# configured with SOFA_FLOATING_POINT_TYPE=double.
if(NOT "${SOFA_FLOATING_POINT_TYPE}" STREQUAL "double")
    message(FATAL_ERROR
        "SofaCHOLMOD requires SOFA_FLOATING_POINT_TYPE=double (CHOLMOD only "
        "supports double precision), but it is currently set to "
        "'${SOFA_FLOATING_POINT_TYPE}'. Set SOFA_FLOATING_POINT_TYPE=double or "
        "disable this plugin (PLUGIN_SOFACHOLMOD=OFF).")
endif()

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
sofa_find_package(Eigen3 REQUIRED)
sofa_find_package(CHOLMOD REQUIRED)

set(SOFACHOLMOD_SOURCE_DIR "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFACHOLMOD_SOURCE_DIR}/config.h.in
    ${SOFACHOLMOD_SOURCE_DIR}/init.h
    ${SOFACHOLMOD_SOURCE_DIR}/EigenCholmodSupernodalLLT.h
)

set(SOURCE_FILES
    ${SOFACHOLMOD_SOURCE_DIR}/init.cpp
    ${SOFACHOLMOD_SOURCE_DIR}/EigenCholmodSupernodalLLT.cpp
)

set(TARGET_NAME ${PROJECT_NAME})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC Sofa.Config Sofa.Simulation.Core)
target_link_libraries(${TARGET_NAME} PUBLIC Sofa.Component.LinearSolver.Direct)
target_link_libraries(${PROJECT_NAME} PUBLIC ${CHOLMOD_LIBRARIES})
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${CHOLMOD_INCLUDE_DIRS})

# Needed to resolve the BLAS backend's runtime thread-count setter via dlsym
# (empty on Windows, where GetProcAddress from kernel32 is used instead).
target_link_libraries(${PROJECT_NAME} PRIVATE ${CMAKE_DL_LIBS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

include(CMakeDependentOption)
cmake_dependent_option(SOFACHOLMOD_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFACHOLMOD_BUILD_TESTS)
    add_subdirectory(tests)
endif()
