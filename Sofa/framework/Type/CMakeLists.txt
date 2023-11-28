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
    ${SOFATYPESRC_ROOT}/Mat_solve_LCP.h
    ${SOFATYPESRC_ROOT}/trait/is_container.h
    ${SOFATYPESRC_ROOT}/trait/is_vector.h
    ${SOFATYPESRC_ROOT}/trait/Rebind.h
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
    ${SOFATYPESRC_ROOT}/fixed_array.h
    ${SOFATYPESRC_ROOT}/fixed_array_algorithms.h
    ${SOFATYPESRC_ROOT}/stable_vector.h
    ${SOFATYPESRC_ROOT}/vector.h
    ${SOFATYPESRC_ROOT}/vector_T.h
    ${SOFATYPESRC_ROOT}/vector_T.inl
    ${SOFATYPESRC_ROOT}/vector_Real.h
    ${SOFATYPESRC_ROOT}/vector_Integral.h
    ${SOFATYPESRC_ROOT}/vector_String.h
    ${SOFATYPESRC_ROOT}/vector_algebra.h
    ${SOFATYPESRC_ROOT}/vector_algorithm.h
    ${SOFATYPESRC_ROOT}/vector_device.h
    ${SOFATYPESRC_ROOT}/SVector.h
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
    ${SOFATYPESRC_ROOT}/fixed_array.cpp
    ${SOFATYPESRC_ROOT}/vector.cpp
    ${SOFATYPESRC_ROOT}/vector_Real.cpp
    ${SOFATYPESRC_ROOT}/vector_Integral.cpp
    ${SOFATYPESRC_ROOT}/vector_String.cpp
    ${SOFATYPESRC_ROOT}/SVector.cpp
    ${SOFATYPESRC_ROOT}/isRigidType.h
)

sofa_find_package(Sofa.Config REQUIRED)

## Boost (1.54.0 or higher) is now mandatory.
set(BOOST_MIN_VERSION "1.54.0")
set(Boost_NO_BOOST_CMAKE TRUE)
sofa_find_package(Boost ${BOOST_MIN_VERSION} QUIET)
if(NOT Boost_FOUND)
    if(WIN32)
        message(FATAL_ERROR "Boost (${BOOST_MIN_VERSION} or higher) is mandatory.\n"
            "Note that Boost autodetection on Windows was removed.\n"
            "Please follow CMake documentation to find Boost on your system.\n"
            "Hint: set BOOST_ROOT to your Boost location."
            )
    else()
        message(FATAL_ERROR "Boost (${BOOST_MIN_VERSION} or higher) is mandatory.")
    endif()
endif()
sofa_find_package(Boost QUIET REQUIRED)
sofa_find_package(Sofa.Compat QUIET BOTH_SCOPES)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)
target_link_libraries(${PROJECT_NAME} PUBLIC Boost::boost)
if(Sofa.Compat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Compat)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_TYPE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_TYPE_BUILD_TESTS)
    add_subdirectory(test)
endif()
