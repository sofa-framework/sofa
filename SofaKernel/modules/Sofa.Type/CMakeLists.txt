cmake_minimum_required(VERSION 3.12)
project(Sofa.Type LANGUAGES CXX)

set(SOFATYPESRC_ROOT "src/sofa/type")

set(HEADER_FILES
    ${SOFATYPESRC_ROOT}/config.h.in
    ${SOFATYPESRC_ROOT}/init.h
    ${SOFATYPESRC_ROOT}/Vec.h
    ${SOFATYPESRC_ROOT}/Mat.h
    ${SOFATYPESRC_ROOT}/MatSym.h
    ${SOFATYPESRC_ROOT}/Mat_solve_Cholesky.h
    ${SOFATYPESRC_ROOT}/Mat_solve_LU.h
    ${SOFATYPESRC_ROOT}/Mat_solve_SVD.h
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array.h
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array_algorithms.h
)

set(SOURCE_FILES
    ${SOFATYPESRC_ROOT}/init.cpp
    ${SOFATYPESRC_ROOT}/Vec.cpp
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array.cpp
)

# Compatibility files
set (COMPATSRC_ROOT "compat/")
set(COMPAT_HEADER_FILES
    ${COMPATSRC_ROOT}/sofa/defaulttype/Vec.h
    ${COMPATSRC_ROOT}/sofa/defaulttype/Mat.h
    ${COMPATSRC_ROOT}/sofa/defaulttype/MatSym.h
    ${COMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_Cholesky.h
    ${COMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_LU.h
    ${COMPATSRC_ROOT}/sofa/defaulttype/Mat_solve_SVD.h
    ${COMPATSRC_ROOT}/sofa/helper/fixed_array.h
    ${COMPATSRC_ROOT}/sofa/helper/fixed_array_algorithms.h
)

find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${COMPAT_HEADER_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)

target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${COMPATSRC_ROOT}>
                           $<INSTALL_INTERFACE:include/SofaFramework/compat>)
source_group("compat" FILES ${COMPAT_HEADER_FILES})

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )

## Tests
# cmake_dependent_option(SOFA_TYPE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFA_TYPE_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()
