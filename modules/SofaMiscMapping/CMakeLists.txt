cmake_minimum_required(VERSION 3.12)
project(SofaMiscMapping LANGUAGES CXX)

set(SOFAMISCMAPPING_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAMISCMAPPING_SRC}/config.h.in
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.h
    )

set(SOURCE_FILES
    ${SOFAMISCMAPPING_SRC}/initSofaMiscMapping.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCMAPPING_SRC}/BarycentricMappingRigid.h
    ${SOFAMISCMAPPING_SRC}/BarycentricMappingRigid.inl
    ${SOFAMISCMAPPING_SRC}/BeamLinearMapping.h
    ${SOFAMISCMAPPING_SRC}/BeamLinearMapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMappingOperation.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMulti2Mapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMulti2Mapping.inl
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/DeformableOnRigidFrameMapping.h
    ${SOFAMISCMAPPING_SRC}/DeformableOnRigidFrameMapping.inl
    ${SOFAMISCMAPPING_SRC}/DistanceFromTargetMapping.h
    ${SOFAMISCMAPPING_SRC}/DistanceFromTargetMapping.inl
    ${SOFAMISCMAPPING_SRC}/DistanceMapping.h
    ${SOFAMISCMAPPING_SRC}/DistanceMapping.inl
    ${SOFAMISCMAPPING_SRC}/IdentityMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/IdentityMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/SquareDistanceMapping.h
    ${SOFAMISCMAPPING_SRC}/SquareDistanceMapping.inl
    ${SOFAMISCMAPPING_SRC}/SquareMapping.h
    ${SOFAMISCMAPPING_SRC}/SquareMapping.inl
    ${SOFAMISCMAPPING_SRC}/SubsetMultiMapping.h
    ${SOFAMISCMAPPING_SRC}/SubsetMultiMapping.inl
    ${SOFAMISCMAPPING_SRC}/TubularMapping.h
    ${SOFAMISCMAPPING_SRC}/TubularMapping.inl
    ${SOFAMISCMAPPING_SRC}/VoidMapping.h
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCMAPPING_SRC}/BarycentricMappingRigid.cpp
    ${SOFAMISCMAPPING_SRC}/BeamLinearMapping.cpp
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMapping.cpp
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMulti2Mapping.cpp
    ${SOFAMISCMAPPING_SRC}/CenterOfMassMultiMapping.cpp
    ${SOFAMISCMAPPING_SRC}/DeformableOnRigidFrameMapping.cpp
    ${SOFAMISCMAPPING_SRC}/DistanceFromTargetMapping.cpp
    ${SOFAMISCMAPPING_SRC}/DistanceMapping.cpp
    ${SOFAMISCMAPPING_SRC}/IdentityMultiMapping.cpp
    ${SOFAMISCMAPPING_SRC}/SquareDistanceMapping.cpp
    ${SOFAMISCMAPPING_SRC}/SquareMapping.cpp
    ${SOFAMISCMAPPING_SRC}/SubsetMultiMapping.cpp
    ${SOFAMISCMAPPING_SRC}/TubularMapping.cpp
    ${SOFAMISCMAPPING_SRC}/VoidMapping.cpp
    )

sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaRigid REQUIRED) 
sofa_find_package(SofaGeneralSimpleFem REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaRigid SofaGeneralSimpleFem)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFAMISCMAPPING_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFAMISCMAPPING_BUILD_TESTS)
    enable_testing()
    add_subdirectory(SofaMiscMapping_test)
endif()