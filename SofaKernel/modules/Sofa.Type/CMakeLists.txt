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
)

sofa_find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaFramework
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaFramework"
    )
