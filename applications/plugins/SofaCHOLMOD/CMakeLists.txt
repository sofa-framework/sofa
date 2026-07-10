cmake_minimum_required(VERSION 3.22)
project(SofaCHOLMOD LANGUAGES CXX)

find_package(Sofa.Config REQUIRED)
sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.Component.LinearSolver.Direct REQUIRED)
sofa_find_package(Eigen3 REQUIRED)
sofa_find_package(CHOLMOD REQUIRED)

get_target_property(SOFACHOLMOD_VERSION Sofa.Config SOFA_VERSION)

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
    PACKAGE_VERSION ${SOFACHOLMOD_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
