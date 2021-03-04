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
    ${SOFATYPESRC_ROOT}/trait/is_container.h
    ${SOFATYPESRC_ROOT}/fwd.h
    ${SOFATYPESRC_ROOT}/Quat.h
    ${SOFATYPESRC_ROOT}/Quat.inl
    ${SOFATYPESRC_ROOT}/DualQuat.h
    ${SOFATYPESRC_ROOT}/DualQuat.inl
    ${SOFATYPESRC_ROOT}/BoundingBox.h
    ${SOFATYPESRC_ROOT}/Color.h
    ${SOFATYPESRC_ROOT}/Frame.h
    ${SOFATYPESRC_ROOT}/Material.h
    ${SOFATYPESRC_ROOT}/PrimitiveGroup.h
    ${SOFATYPESRC_ROOT}/Ray.h
    ${SOFATYPESRC_ROOT}/RGBAColor.h
    ${SOFATYPESRC_ROOT}/RGBAColor_fwd.h
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array.h
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array_algorithms.h
    ${SOFATYPESRC_ROOT}/stdtype/stable_vector.h
    ${SOFATYPESRC_ROOT}/stdtype/vector.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_T.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_T.inl
    ${SOFATYPESRC_ROOT}/stdtype/vector_Real.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_Integral.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_String.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_algebra.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_algorithm.h
    ${SOFATYPESRC_ROOT}/stdtype/vector_device.h
    ${SOFATYPESRC_ROOT}/stdtype/SVector.h
)

set(SOURCE_FILES
    ${SOFATYPESRC_ROOT}/init.cpp
    ${SOFATYPESRC_ROOT}/Vec.cpp
    ${SOFATYPESRC_ROOT}/Quat.cpp
    ${SOFATYPESRC_ROOT}/DualQuat.cpp
    ${SOFATYPESRC_ROOT}/BoundingBox.cpp
    ${SOFATYPESRC_ROOT}/Frame.cpp
    ${SOFATYPESRC_ROOT}/Material.cpp
    ${SOFATYPESRC_ROOT}/PrimitiveGroup.cpp
    ${SOFATYPESRC_ROOT}/RGBAColor.cpp
    ${SOFATYPESRC_ROOT}/stdtype/fixed_array.cpp
    ${SOFATYPESRC_ROOT}/stdtype/vector.cpp
    ${SOFATYPESRC_ROOT}/stdtype/vector_Real.cpp
    ${SOFATYPESRC_ROOT}/stdtype/vector_Integral.cpp
    ${SOFATYPESRC_ROOT}/stdtype/vector_String.cpp
    ${SOFATYPESRC_ROOT}/stdtype/SVector.cpp
)

sofa_find_package(Boost REQUIRED)

sofa_find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)
target_link_libraries(${PROJECT_NAME} PUBLIC Boost::boost)

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
